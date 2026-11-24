"""Fast (<5s) static pre-route feasibility gate for Clay flows.

Moves multi-decade router/LVS/PEX dead-ends forward from ~30-minute gridless
negotiation or post-route failures into an actionable <5s pre-flight verdict:

  * **T1 Source-PDK-LVS (<0.5s)**: device primitive support coverage,
    GF180 MIM Option-A vs Option-B stack compatibility with the selected
    PEX backend (e.g. Magic ``gf180mcuD`` M4/CAPM/CAPDEF/M5), and semantic
    substrate / isolated-well closure (``clay.port_roles`` +
    ``clay.well_isolation``).
  * **T2 Unexempted Pin-Escape, Via-Redundancy Clearance & Corridor Congestion (<2s)**:
    runs :func:`verify_static_terminal_escapability` unconditionally (even when
    ``group_route_contract`` is active), audits ``via_redundancy=N`` landing
    pad clearance against nearby foreign obstacles, and estimates horizontal /
    vertical inter-device corridor track supply vs demand.
  * **T3 Actionable Structured Blockers**: every failure surfaces exact
    ``(net, pin, blocker_bbox_nm, min_expand_nm, reason, suggested_action)``
    with zero silent bypasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import klayout.db as db

Rect = Tuple[int, int, int, int]
LayerKey = Tuple[int, int]

_NM_PER_UM = 1000.0
_ESCAPE_REACH_NM = 1600
_DEFAULT_WELL_SEP_NM = 1400


@dataclass(frozen=True)
class PreRouteBlocker:
    """One actionable static pre-route or LVS/PDK blocking finding."""

    tier: str
    category: str
    net: str
    pin: str
    blocker_bbox_nm: Optional[Rect]
    min_expand_nm: int
    reason: str
    suggested_action: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "category": self.category,
            "net": self.net,
            "pin": self.pin,
            "blocker_bbox_nm": (
                list(self.blocker_bbox_nm)
                if self.blocker_bbox_nm is not None
                else None
            ),
            "min_expand_nm": int(self.min_expand_nm),
            "reason": self.reason,
            "suggested_action": self.suggested_action,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class PreRouteGateResult:
    """Complete T1 + T2 verdict returned by :func:`run_preroute_gate`."""

    passed: bool
    t1_passed: bool
    t2_passed: bool
    elapsed_ms: float
    budget_sec: float
    timed_out: bool = False
    blockers: Tuple[PreRouteBlocker, ...] = ()
    recommended_expand_nm: int = 0
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "t1_passed": self.t1_passed,
            "t2_passed": self.t2_passed,
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "budget_sec": float(self.budget_sec),
            "timed_out": bool(self.timed_out),
            "recommended_expand_nm": int(self.recommended_expand_nm),
            "summary": self.summary,
            "blockers": [b.to_dict() for b in self.blockers],
            "metrics": dict(self.metrics),
        }


# --------------------------------------------------------------------------- #
# Small geometry & metadata helpers
# --------------------------------------------------------------------------- #
def _norm_rect(rect: Sequence[int]) -> Rect:
    x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _rect_gap_nm(a: Rect, b: Rect) -> Tuple[int, int]:
    """Return axis gaps ``(dx, dy)`` in nm between normalized rects (0 if overlap/touch)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0, max(ay0, by0) - min(ay1, by1))
    return dx, dy


def _ceil_grid(value_nm: float, grid_nm: int) -> int:
    g = max(1, int(grid_nm))
    val = int(math.ceil(max(0.0, float(value_nm))))
    if val <= 0:
        return 0
    return ((val + g - 1) // g) * g


def _resolve_pdk_id(pdk: Any, cell_def: Any) -> str:
    if hasattr(pdk, "pdk_id") and getattr(pdk, "pdk_id", None):
        return str(pdk.pdk_id)
    if isinstance(pdk, str) and pdk.strip():
        return pdk.strip()
    target = getattr(cell_def, "target_pdk", None) if cell_def is not None else None
    if target:
        return str(target)
    return "gf180mcu"


def _resolve_tech(pdk: Any, pdk_id: str) -> Any:
    from clay.layout.tech import LayoutTech

    if isinstance(pdk, LayoutTech):
        return pdk
    return LayoutTech.resolve(pdk_id)


def _device_bbox_map_nm(placement: Any) -> Dict[str, Rect]:
    out: Dict[str, Rect] = {}
    devices = getattr(placement, "devices", None)
    if isinstance(devices, Mapping):
        for name, pd in devices.items():
            bbox = getattr(pd, "bbox", None)
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                out[str(name)] = _norm_rect(bbox)
    return out


def _geometry_layers_bbox_nm(
    geom: Any,
    center_um: Tuple[float, float],
) -> Dict[LayerKey, List[Rect]]:
    """Transform one Stamp / CellGeometry's layer polygons to absolute int-nm rects."""
    out: Dict[LayerKey, List[Rect]] = {}
    polys_by_layer = getattr(geom, "polys_by_layer", None) or {}
    if not polys_by_layer:
        return out
    from clay.layout.placed_geometry import placed_point_nm

    for lk, rings in polys_by_layer.items():
        key = (int(lk[0]), int(lk[1]))
        rects = out.setdefault(key, [])
        for ring in rings:
            pts = [placed_point_nm(float(x), float(y), center_um) for x, y in ring]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            rects.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def _union_bbox_nm(rects: Sequence[Rect]) -> Optional[Rect]:
    if not rects:
        return None
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _extract_device_entries(
    cell_def: Any,
    device_records: Optional[Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect unified MOS and passive inventories from ``cell_def`` + ``device_records``."""
    mos_list: List[Dict[str, Any]] = []
    passive_list: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()

    if cell_def is not None:
        for dev in getattr(cell_def, "devices", ()) or ():
            name = str(getattr(dev, "instance_name", "") or getattr(dev, "name", ""))
            if not name:
                continue
            seen_names.add(name)
            cls = str(getattr(dev, "device_class", "") or "").lower()
            pins = dict(getattr(dev, "pins", {}) or {})
            norm_pins = {str(k).lower(): str(v) for k, v in pins.items()}
            mos_list.append({
                "name": name,
                "kind": cls,
                "pins": norm_pins,
                "model": str(
                    getattr(dev, "source_model", "")
                    or getattr(dev, "model", "")
                    or ""
                ),
            })
        for pas in getattr(cell_def, "passives", ()) or ():
            name = str(getattr(pas, "instance_name", "") or getattr(pas, "name", ""))
            if not name:
                continue
            seen_names.add(name)
            cls = str(
                getattr(pas, "device_class", None)
                or getattr(pas, "kind", "")
                or ""
            ).lower()
            pins = dict(getattr(pas, "pins", None) or getattr(pas, "nets", {}) or {})
            norm_pins = {str(k).lower(): str(v) for k, v in pins.items()}
            passive_list.append({
                "name": name,
                "kind": cls,
                "pins": norm_pins,
                "model": str(getattr(pas, "model", "") or ""),
            })

    if device_records:
        for raw_name, rec in device_records.items():
            name = str(raw_name)
            if name in seen_names or not isinstance(rec, Mapping):
                continue
            cls = str(rec.get("kind") or rec.get("device_class") or "").lower()
            pins = dict(rec.get("pins") or rec.get("nets") or {})
            norm_pins = {str(k).lower(): str(v) for k, v in pins.items()}
            entry = {
                "name": name,
                "kind": cls,
                "pins": norm_pins,
                "model": str(rec.get("model") or ""),
            }
            if cls.startswith(("cap", "mim", "res")):
                passive_list.append(entry)
            else:
                mos_list.append(entry)

    return mos_list, passive_list


# --------------------------------------------------------------------------- #
# T1: Source-PDK-LVS precheck (<0.5s)
# --------------------------------------------------------------------------- #
def _check_t1_device_support(
    cell_def: Any,
    pdk_id: str,
    *,
    physical: bool,
    source_artifact: Optional[str],
    dev_bboxes: Mapping[str, Rect],
) -> List[PreRouteBlocker]:
    if cell_def is None:
        return []
    from clay.flow.physical_support import check_device_support

    blockers: List[PreRouteBlocker] = []
    try:
        support = check_device_support(
            cell_def,
            pdk_id,
            physical=physical,
            source_artifact=source_artifact,
        )
    except Exception as exc:
        blockers.append(
            PreRouteBlocker(
                tier="T1",
                category="device_support",
                net="",
                pin="",
                blocker_bbox_nm=None,
                min_expand_nm=0,
                reason=f"check_device_support raised {type(exc).__name__}: {exc}",
                suggested_action=f"Verify primitive declarations and PDK bindings for {pdk_id!r}",
            )
        )
        return blockers

    for chk in getattr(support, "checks", ()) or ():
        if getattr(chk, "status", "pass") in {"fail", "unsupported", "error"}:
            detail = dict(getattr(chk, "detail", None) or {})
            dev_name = str(
                detail.get("device")
                or detail.get("instance")
                or detail.get("name")
                or ""
            )
            blockers.append(
                PreRouteBlocker(
                    tier="T1",
                    category="device_support",
                    net=str(detail.get("net") or ""),
                    pin=dev_name,
                    blocker_bbox_nm=dev_bboxes.get(dev_name),
                    min_expand_nm=0,
                    reason=f"{chk.code}: {chk.message}",
                    suggested_action=(
                        f"Provide a supported {pdk_id} primitive or binding for "
                        f"{dev_name or chk.name}"
                    ),
                    detail={"code": chk.code, "check_name": chk.name, **detail},
                )
            )
    return blockers


def _resolve_pex_and_cap_option(
    *,
    pex_backend: Optional[str],
    cap_option: Optional[str],
    cell_def: Any,
    effective_policy: Any,
    intent: Any,
) -> Tuple[Optional[str], Optional[str]]:
    pex = pex_backend
    if pex is None and effective_policy is not None:
        pex = getattr(effective_policy, "pex_backend", None)
    if pex is None and cell_def is not None:
        flow_decl = getattr(cell_def, "flow_decl", None)
        pex = getattr(flow_decl, "pex_backend", None) or getattr(
            cell_def, "pex_backend", None
        )
    pex_norm = str(pex).strip().lower() if pex else None

    opt = cap_option
    if opt is None and effective_policy is not None:
        opt = getattr(effective_policy, "cap_option", None) or getattr(
            effective_policy, "mim_option", None
        )
    if opt is None and intent is not None:
        opt = getattr(intent, "cap_option", None) or getattr(intent, "mim_option", None)
    if opt is None and cell_def is not None:
        flow_decl = getattr(cell_def, "flow_decl", None)
        opt = (
            getattr(flow_decl, "cap_option", None)
            or getattr(flow_decl, "mim_option", None)
            or getattr(cell_def, "cap_option", None)
        )
    opt_norm = str(opt).strip().lower().replace("-", "_") if opt else None
    if opt_norm in {"a", "opt_a"}:
        opt_norm = "option_a"
    elif opt_norm in {"b", "opt_b"}:
        opt_norm = "option_b"
    return pex_norm, opt_norm


def _rect_covered_by_any(target: Rect, candidates: Sequence[Rect]) -> bool:
    if not candidates:
        return False
    reg_t = db.Region(db.Box(*target))
    reg_c = db.Region()
    for rect in candidates:
        reg_c.insert(db.Box(*rect))
    reg_c.merge()
    return (reg_t - reg_c).is_empty()


def _check_t1_gf180_mim_stack(
    *,
    pdk_id: str,
    tech: Any,
    passives: Sequence[Mapping[str, Any]],
    geometry: Optional[Mapping[str, Any]],
    placement_um: Optional[Mapping[str, Tuple[float, float]]],
    dev_bboxes: Mapping[str, Rect],
    pex_backend: Optional[str],
    cap_option: Optional[str],
) -> List[PreRouteBlocker]:
    """Validate GF180 MIM Option-A vs Option-B consistency against the active PEX backend."""
    if pdk_id != "gf180mcu":
        return []

    cap_devs = [
        p
        for p in passives
        if p["kind"].startswith(("cap", "mim")) or "mim" in p["model"].lower()
    ]
    capm_key: LayerKey = (75, 0)
    capdef_key: LayerKey = (117, 5)
    try:
        m4_key: Optional[LayerKey] = tuple(tech.metal_key(4))
        m2_key: Optional[LayerKey] = tuple(tech.metal_key(2))
        m3_key: Optional[LayerKey] = tuple(tech.metal_key(3))
    except Exception:
        m4_key = (46, 0)
        m2_key = (36, 0)
        m3_key = (42, 0)

    # Inspect built geometry first (ground-truth artwork if materialized).
    geom_cap_hits: List[Tuple[str, Rect, bool, bool, Optional[str]]] = []
    if geometry:
        for name, stamp in geometry.items():
            center = (
                tuple(placement_um[name])
                if placement_um and name in placement_um
                else (0.0, 0.0)
            )
            layers = _geometry_layers_bbox_nm(stamp, center)
            capm_rects = layers.get(capm_key, [])
            if not capm_rects:
                continue
            capdef_rects = layers.get(capdef_key, [])
            m4_rects = layers.get(m4_key, []) if m4_key else []
            m2_rects = layers.get(m2_key, []) if m2_key else []
            m3_rects = layers.get(m3_key, []) if m3_key else []
            is_opt_b = all(
                _rect_covered_by_any(r, m4_rects) and _rect_covered_by_any(r, capdef_rects)
                for r in capm_rects
            )
            is_opt_a = bool(m2_rects and m3_rects and not is_opt_b)
            inferred = "option_b" if is_opt_b else ("option_a" if is_opt_a else None)
            geom_cap_hits.append((
                str(name),
                _union_bbox_nm(capm_rects) or dev_bboxes.get(str(name)),
                is_opt_b,
                is_opt_a,
                inferred,
            ))

    if not cap_devs and not geom_cap_hits:
        return []

    blockers: List[PreRouteBlocker] = []

    # Check explicit Option-B vs incomplete geometry
    if cap_option == "option_b":
        for name, bbox, is_opt_b, _is_opt_a, _inferred in geom_cap_hits:
            if not is_opt_b:
                pins = next((c["pins"] for c in cap_devs if c["name"] == name), {})
                net_str = ",".join(sorted({v for v in pins.values() if v}))
                blockers.append(
                    PreRouteBlocker(
                        tier="T1",
                        category="gf180_mim_stack_mismatch",
                        net=net_str,
                        pin=name,
                        blocker_bbox_nm=bbox,
                        min_expand_nm=0,
                        reason=(
                            f"GF180_MIM_OPTION_B_INCOMPLETE: cap {name!r} has CAPM(75/0) "
                            "artwork not covered by M4(46/0) and CAPDEF(117/5)"
                        ),
                        suggested_action=(
                            "Materialize MIM Option-B stamp using M4 bottom, CAPM(75/0) "
                            "with CAPDEF(117/5), and M5 top access"
                        ),
                        detail={"cap": name, "cap_option": cap_option, "pex_backend": pex_backend},
                    )
                )

    # Check Magic backend compatibility against Option-A / default GF180 MIM builder
    if pex_backend == "magic":
        if geom_cap_hits:
            for name, bbox, is_opt_b, _is_opt_a, inferred in geom_cap_hits:
                if not is_opt_b:
                    pins = next((c["pins"] for c in cap_devs if c["name"] == name), {})
                    net_str = ",".join(sorted({v for v in pins.values() if v}))
                    blockers.append(
                        PreRouteBlocker(
                            tier="T1",
                            category="gf180_mim_stack_mismatch",
                            net=net_str,
                            pin=name,
                            blocker_bbox_nm=bbox,
                            min_expand_nm=0,
                            reason=(
                                f"GF180_MIM_STACK_MISMATCH: Magic gf180mcuD requires Option-B MIM "
                                f"(M4(46/0)+CAPM(75/0)+CAPDEF(117/5)+M5), whereas cap {name!r} uses "
                                f"{inferred or 'option_a'} (lower stack)"
                            ),
                            suggested_action=(
                                "Switch GF180 MIM builder/flow to Option-B (M4/CAPM/CAPDEF/M5) "
                                "or select pex_backend='geometric'"
                            ),
                            detail={"cap": name, "detected": inferred, "pex_backend": pex_backend},
                        )
                    )
        else:
            # Source-only check before geometry materialization: GF180 passive_device_cell defaults to Option-A
            effective_opt = cap_option or "option_a"
            if effective_opt != "option_b":
                for cap in cap_devs:
                    net_str = ",".join(sorted({v for v in cap["pins"].values() if v}))
                    blockers.append(
                        PreRouteBlocker(
                            tier="T1",
                            category="gf180_mim_stack_mismatch",
                            net=net_str,
                            pin=cap["name"],
                            blocker_bbox_nm=dev_bboxes.get(cap["name"]),
                            min_expand_nm=0,
                            reason=(
                                f"GF180_MIM_STACK_MISMATCH: Magic gf180mcuD PEX backend requires "
                                f"cap_option='option_b' (M4/CAPM/CAPDEF/M5), not {effective_opt!r}"
                            ),
                            suggested_action=(
                                "Configure cap_option='option_b' for Magic PEX extraction, "
                                "or use pex_backend='geometric' for Option-A (M2/M3/Via2)"
                            ),
                            detail={"cap": cap["name"], "cap_option": effective_opt, "pex_backend": pex_backend},
                        )
                    )

    return blockers


def _resolve_substrate_and_supply_nets(cell_def: Any) -> Tuple[str, Set[str], str, Set[str]]:
    from clay.port_roles import (
        declared_positive_supply_ports,
        declared_substrate_ports,
        resolve_positive_supply_net,
        resolve_substrate_net,
    )

    port_roles = getattr(cell_def, "port_roles", None) if cell_def is not None else None
    sub_ports = set(declared_substrate_ports(port_roles))
    pos_ports = set(declared_positive_supply_ports(port_roles))

    sub_fallback = "gnd"
    pos_fallback = "vdd"
    if cell_def is not None:
        port_names = [
            str(getattr(p, "name", "") or "")
            for p in (getattr(cell_def, "ports", ()) or ())
        ]
        for cand in ("gnd", "vss", "avss", "dgnd", "sub", "vnb"):
            hit = next((pn for pn in port_names if pn.lower() == cand), None)
            if hit:
                sub_fallback = hit
                break
        for cand in ("vdd", "vdda", "avdd", "vpwr", "vcc"):
            hit = next((pn for pn in port_names if pn.lower() == cand), None)
            if hit:
                pos_fallback = hit
                break

    if cell_def is not None:
        sub_net = resolve_substrate_net(cell_def, fallback=sub_fallback)
        pos_net = resolve_positive_supply_net(cell_def, fallback=pos_fallback)
    else:
        sub_net, pos_net = sub_fallback, pos_fallback

    if not sub_ports:
        sub_ports = {sub_net, "gnd", "vss", "sub", "vss!", "gnd!"}
    if not pos_ports:
        pos_ports = {pos_net, "vdd", "vdda", "vdd!", "vpwr"}
    return sub_net, sub_ports, pos_net, pos_ports


def _well_min_space_nm(tech: Any) -> int:
    try:
        rules = tech.drc_rules() or {}
        nw = rules.get("nwell_min_space") or rules.get("nwell_space")
        if isinstance(nw, dict):
            if nw.get("value_nm") is not None:
                return int(nw["value_nm"])
            if nw.get("value_um") is not None:
                return int(round(float(nw["value_um"]) * _NM_PER_UM))
    except Exception:
        pass
    try:
        iw = getattr(tech.pdk, "implant_well_rules", None)
        enc = int(getattr(iw, "nwell_pcomp_enclosure", 430) or 430)
        return max(1000, 2 * enc + 500)
    except Exception:
        return _DEFAULT_WELL_SEP_NM


def _check_t1_well_and_substrate_closure(
    *,
    cell_def: Any,
    intent: Any,
    pdk_id: str,
    tech: Any,
    mos_devices: Sequence[Mapping[str, Any]],
    dev_bboxes: Mapping[str, Rect],
) -> List[PreRouteBlocker]:
    """Validate NMOS isolated-well coverage and PMOS private-well non-sharing."""
    if not mos_devices:
        return []

    from clay.pdk import get_pdk
    from clay.well_isolation import (
        declared_well_isolated_devices,
        deep_well_lvs_model_overrides,
    )

    blockers: List[PreRouteBlocker] = []
    sub_net, sub_ports, pos_net, pos_ports = _resolve_substrate_and_supply_nets(cell_def)
    isolated_devs = (
        set(declared_well_isolated_devices(cell_def))
        if cell_def is not None
        else set()
    )
    dnw_overrides: Dict[str, str] = {}
    if cell_def is not None:
        try:
            dnw_overrides = deep_well_lvs_model_overrides(cell_def, pdk_id)
        except Exception:
            dnw_overrides = {}

    try:
        adapter = get_pdk(pdk_id)
        pdk_dnw = getattr(adapter, "deep_nwell_override", None)
    except Exception:
        pdk_dnw = None

    mos_by_name = {d["name"]: d for d in mos_devices}
    sub_ports_low = {s.lower().rstrip("!") for s in sub_ports}
    pos_ports_low = {p.lower().rstrip("!") for p in pos_ports}

    def _mos_polarity(dev_entry: Mapping[str, Any]) -> Optional[str]:
        """Return 'n' for NMOS, 'p' for PMOS, or None for non-MOS / unknown native."""
        kind_s = str(dev_entry.get("kind") or "").strip().lower()
        model_s = str(dev_entry.get("model") or "").strip().lower()
        if kind_s == "native" or not kind_s:
            if any(tok in model_s for tok in ("pfet", "pmos", "pch")):
                return "p"
            if any(tok in model_s for tok in ("nfet", "nmos", "nch")):
                return "n"
            return None
        if "pmos" in kind_s or "pfet" in kind_s or "pch" in kind_s or kind_s.startswith("p"):
            return "p"
        if "nmos" in kind_s or "nfet" in kind_s or "nch" in kind_s or kind_s.startswith("n"):
            return "n"
        return None

    # 1. NMOS bulk != substrate -> requires @well_isolated + PDK DNW support
    for dev in mos_devices:
        if _mos_polarity(dev) != "n":
            continue
        bulk = str(dev["pins"].get("b") or dev["pins"].get("body") or "").strip()
        if not bulk:
            continue
        if bulk.lower().rstrip("!") in sub_ports_low:
            continue
        name = dev["name"]
        if name not in isolated_devs:
            blockers.append(
                PreRouteBlocker(
                    tier="T1",
                    category="substrate_isolation",
                    net=bulk,
                    pin=f"{name}.b",
                    blocker_bbox_nm=dev_bboxes.get(name),
                    min_expand_nm=0,
                    reason=(
                        f"UNISOLATED_NMOS_BODY: NMOS {name!r} bulk tied to {bulk!r} != "
                        f"substrate {sub_net!r} without @well_isolated({name})"
                    ),
                    suggested_action=(
                        f"Add `@well_isolated({name});` inside `annotate` block or "
                        f"tie {name}.b to substrate {sub_net!r}"
                    ),
                    detail={"device": name, "bulk": bulk, "substrate": sub_net},
                )
            )
        elif pdk_dnw is None and name not in dnw_overrides:
            blockers.append(
                PreRouteBlocker(
                    tier="T1",
                    category="substrate_isolation",
                    net=bulk,
                    pin=f"{name}.b",
                    blocker_bbox_nm=dev_bboxes.get(name),
                    min_expand_nm=0,
                    reason=(
                        f"UNSUPPORTED_DEEP_NWELL: NMOS {name!r} declares @well_isolated "
                        f"for body {bulk!r}, but PDK {pdk_id!r} provides no deep_nwell_override"
                    ),
                    suggested_action=(
                        f"Tie {name}.b to substrate {sub_net!r} or supply DNW override on {pdk_id!r}"
                    ),
                    detail={"device": name, "bulk": bulk, "pdk": pdk_id},
                )
            )

    # 2. PMOS private well closure (guard-ring well mixing or too-close private wells)
    pmos_devs = [
        d
        for d in mos_devices
        if _mos_polarity(d) == "p"
    ]
    pmos_bulk: Dict[str, str] = {}
    for dev in pmos_devs:
        b = str(dev["pins"].get("b") or dev["pins"].get("body") or "").strip()
        if b:
            pmos_bulk[dev["name"]] = b

    # 2a. Same @guard_ring mixing different well potentials
    rings = tuple(getattr(intent, "guard_rings", ()) or ()) if intent is not None else ()
    for gr in rings:
        members = [str(m) for m in (getattr(gr, "members", ()) or ())]
        gr_well = str(getattr(gr, "net", "") or "").strip()
        bulks_in_ring: Dict[str, List[str]] = {}
        for m in members:
            dev = mos_by_name.get(m)
            if dev is None or m not in pmos_bulk:
                continue
            b_net = pmos_bulk[m]
            bulks_in_ring.setdefault(b_net, []).append(m)
        if len(bulks_in_ring) > 1:
            private_members = [
                m
                for b_net, ms in bulks_in_ring.items()
                if b_net.lower().rstrip("!") not in pos_ports_low
                for m in ms
            ]
            bboxes = [dev_bboxes[m] for m in members if m in dev_bboxes]
            blockers.append(
                PreRouteBlocker(
                    tier="T1",
                    category="private_well_conflict",
                    net=",".join(sorted(bulks_in_ring)),
                    pin=",".join(private_members or members),
                    blocker_bbox_nm=_union_bbox_nm(bboxes),
                    min_expand_nm=_well_min_space_nm(tech),
                    reason=(
                        f"PRIVATE_NWELL_RING_CONFLICT: @guard_ring encloses PMOS devices "
                        f"with distinct N-well nets {dict(bulks_in_ring)}"
                    ),
                    suggested_action=(
                        "Split private-well PMOS devices into separate @guard_ring / well islands "
                        "separated by NWELL spacing"
                    ),
                    detail={"ring_net": gr_well, "bulks": bulks_in_ring},
                )
            )

    # 2b. Placed PMOS devices with private well (bulk != pos_net) too close to foreign PMOS N-well
    well_sep_nm = _well_min_space_nm(tech)
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    grid_nm = max(5, int(pdk_manufacturing_grid_nm(tech)))
    for i, a in enumerate(pmos_devs):
        na = a["name"]
        ba = pmos_bulk.get(na, "")
        if not ba or na not in dev_bboxes:
            continue
        ba_is_private = ba.lower().rstrip("!") not in pos_ports_low
        for b in pmos_devs[i + 1 :]:
            nb = b["name"]
            bb = pmos_bulk.get(nb, "")
            if not bb or nb not in dev_bboxes or ba.lower() == bb.lower():
                continue
            bb_is_private = bb.lower().rstrip("!") not in pos_ports_low
            if not (ba_is_private or bb_is_private):
                continue
            dx, dy = _rect_gap_nm(dev_bboxes[na], dev_bboxes[nb])
            actual_gap = max(dx, dy)
            if actual_gap < well_sep_nm:
                expand = _ceil_grid(well_sep_nm - actual_gap, grid_nm)
                blockers.append(
                    PreRouteBlocker(
                        tier="T1",
                        category="private_well_conflict",
                        net=f"{ba},{bb}",
                        pin=f"{na}.b,{nb}.b",
                        blocker_bbox_nm=_union_bbox_nm([dev_bboxes[na], dev_bboxes[nb]]),
                        min_expand_nm=expand,
                        reason=(
                            f"PRIVATE_NWELL_PROXIMITY: PMOS {na}(b={ba}) and {nb}(b={bb}) "
                            f"gap={actual_gap}nm < well separation {well_sep_nm}nm"
                        ),
                        suggested_action=(
                            f"Separate {na} and {nb} by >= {expand}nm or isolate private N-well"
                        ),
                        detail={
                            "dev_a": na,
                            "bulk_a": ba,
                            "dev_b": nb,
                            "bulk_b": bb,
                            "gap_nm": actual_gap,
                            "required_well_sep_nm": well_sep_nm,
                        },
                    )
                )

    return blockers


# --------------------------------------------------------------------------- #
# T2: Router world preparation + unexempted escape + via_redundancy + tracks
# --------------------------------------------------------------------------- #
def _populate_gridless_static_world(
    *,
    tech: Any,
    pins_by_net: Mapping[str, Sequence[Any]],
    obstacles: Sequence[Any],
    constraints: Sequence[Any],
    routing_envelope: Any,
    via_redundancy: int,
    budget_sec: float,
    engine: Any = None,
) -> Tuple[Any, Dict[str, List[Tuple[int, int, int]]], Dict[str, List[Any]], Dict[str, List[int]]]:
    """Build or reuse a lightweight GridlessEngine static world without routing."""
    from clay.layout.route.constraints import LayerSet, Width
    from clay.layout.route.deck_isolation import DeckConnectStack, DeckConnection
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.engine_gridless import GridlessEngine, _Tech
    from clay.layout.route.engine_maze import _via_cut_size_nm, _via_name_for
    from clay.layout.route.gridless_types import (
        _Book,
        _OwnedTerminalAtom,
        _SpatialIndex,
        _norm as _grid_norm,
        _pin_xy_nm,
        _via_cut_space_nm,
    )
    from clay.layout.model import ROUTING_ABSTRACTION_KINDS
    from clay.layout.route.via_model import via_enclosure_nm

    g = engine
    if (
        g is not None
        and isinstance(g, GridlessEngine)
        and getattr(g, "_tech", None) is not None
        and getattr(g, "_obs_idx", None) is not None
    ):
        g_tech = g._tech
        net_nodes: Dict[str, List[Tuple[int, int, int]]] = {}
        net_node_pins: Dict[str, List[Any]] = {}
        net_layers: Dict[str, List[int]] = {}
        all_gis = list(range(g._n_layers))
        layer_map = {
            c.net: [
                gi
                for gi in all_gis
                if g_tech.resolve_layer_to_pdk(g_tech.gi_to_metal_key(gi)) in c.layers
            ]
            for c in constraints
            if isinstance(c, LayerSet)
        }
        for net, plist in pins_by_net.items():
            allowed = layer_map.get(net, all_gis) or all_gis
            net_layers[net] = list(allowed)
            nodes_l: List[Tuple[int, int, int]] = []
            pins_l: List[Any] = []
            for p in plist:
                px, py = _pin_xy_nm(p)
                gi = g_tech.resolve_layer_to_gi(p.layer)
                if gi is None:
                    gi = allowed[0]
                nodes_l.append((px, py, int(gi)))
                pins_l.append(p)
            net_nodes[net] = nodes_l
            net_node_pins[net] = pins_l
        return g, net_nodes, net_node_pins, net_layers

    g = GridlessEngine(
        time_budget_sec=max(0.25, float(budget_sec)),
        max_iters=0,
        via_redundancy=max(1, int(via_redundancy)),
    )
    t = _Tech(tech)
    n_layers = t.n_layers
    g._tech = t
    g._n_layers = n_layers
    g._max_gi = n_layers - 1
    g._manufacturing_grid_nm = max(1, int(pdk_manufacturing_grid_nm(tech)))
    g._watchdog_deadline = time.monotonic() + max(0.25, float(budget_sec))

    half_w: List[int] = []
    space: List[int] = []
    for gi in range(n_layers):
        pdk_idx = t._gi_to_pdk[gi]
        half_w.append(max(1, int(t.min_width_nm[pdk_idx]) // 2))
        space.append(int(t.min_space_nm[pdk_idx]))
    g._half_w = half_w
    g._space = space

    net_half_w, widened, cons_by_net = g._width_half_table(constraints, half_w, n_layers)
    g._net_half_w = net_half_w
    g._widened_nets = widened
    g._width_cons_by_net = cons_by_net

    via_cut: Dict[int, int] = {}
    via_space: Dict[int, int] = {}
    via_pitch: Dict[int, int] = {}
    via_pad_half: Dict[int, Tuple[int, int]] = {}
    via_pad_half1: Dict[int, Tuple[int, int]] = {}
    vr = max(1, int(via_redundancy))
    for lo in range(n_layers - 1):
        pdk_lo = t._gi_to_pdk[lo]
        vname = _via_name_for(t, lo)
        cut = int(_via_cut_size_nm(t, lo) or t.min_width_nm[pdk_lo])
        vsp = int(_via_cut_space_nm(tech, vname, space[lo]))
        eb, ea = via_enclosure_nm(tech, vname)
        pitch = cut + vsp
        if pitch % 2:
            pitch += 1
        arr_span = (vr - 1) * pitch if vr > 1 else 0
        bh = max(half_w[lo], (arr_span + cut + 1) // 2 + int(eb))
        ah = max(half_w[lo + 1], (arr_span + cut + 1) // 2 + int(ea))
        bh1 = max(half_w[lo], (cut + 1) // 2 + int(eb))
        ah1 = max(half_w[lo + 1], (cut + 1) // 2 + int(ea))
        via_cut[lo] = cut
        via_space[lo] = vsp
        via_pitch[lo] = pitch
        via_pad_half[lo] = (bh, ah)
        via_pad_half1[lo] = (bh1, ah1)
    g._via_cut = via_cut
    g._via_space = via_space
    g._via_pitch = via_pitch
    g._via_pad_half = via_pad_half
    g._via_pad_half1 = via_pad_half1

    obs_by_gi: Dict[int, List[Tuple[Optional[str], Rect]]] = {gi: [] for gi in range(n_layers)}
    obs_idx: Dict[int, _SpatialIndex] = {gi: _SpatialIndex() for gi in range(n_layers)}
    literal_planar_idx: Dict[int, _SpatialIndex] = {gi: _SpatialIndex() for gi in range(n_layers)}
    abstract_rects: Dict[int, Set[Rect]] = {}
    owned_atoms: List[_OwnedTerminalAtom] = []
    fixed_literal_rects: List[Tuple[LayerKey, Optional[str], Rect]] = []
    fixed_contact_rects: List[Rect] = []

    try:
        contact_lk: Optional[LayerKey] = tuple(tech.pdk.contact_key())
    except Exception:
        contact_lk = None

    all_rects_for_bounds: List[Rect] = []
    for obs in obstacles:
        rect = _grid_norm((
            int(round(float(obs.x0) * _NM_PER_UM)),
            int(round(float(obs.y0) * _NM_PER_UM)),
            int(round(float(obs.x1) * _NM_PER_UM)),
            int(round(float(obs.y1) * _NM_PER_UM)),
        ))
        owner = getattr(obs, "owner_net", None)
        kind = str(getattr(obs, "kind", "device") or "device")
        lk = (int(obs.layer[0]), int(obs.layer[1]))
        all_rects_for_bounds.append(rect)

        if contact_lk is not None and lk == contact_lk:
            fixed_contact_rects.append(rect)
            continue

        gi = t.resolve_layer_to_gi(lk)
        if gi is not None:
            obs_by_gi[gi].append((owner, rect))
            obs_idx[gi].add(owner, rect)
            if kind in ROUTING_ABSTRACTION_KINDS:
                abstract_rects.setdefault(gi, set()).add(rect)
            else:
                literal_planar_idx[gi].add(owner, rect)
                fixed_literal_rects.append((lk, owner, rect))
            tref = getattr(obs, "terminal_conductor_ref", None)
            if owner is not None and tref:
                owned_atoms.append(
                    _OwnedTerminalAtom(
                        gi=int(gi),
                        owner=str(owner),
                        rect=rect,
                        kind=kind,
                        terminal_conductor_ref=str(tref),
                    )
                )
        else:
            fixed_literal_rects.append((lk, owner, rect))

    g._obs_by_gi = obs_by_gi
    g._obs_idx = obs_idx
    g._literal_planar_idx = literal_planar_idx
    g._abstract_rects = abstract_rects
    g._obs_cuts = {}
    g._obs_provenance = {}
    g._obs_cut_provenance = {}
    g._owned_terminal_atoms = tuple(owned_atoms)
    g._owned_terminal_group_indexes = {}
    g._fixed_deck_literal_rects = tuple(fixed_literal_rects)
    g._fixed_contact_rects = tuple(fixed_contact_rects)
    g._rail_by_net = {}
    g._routing_envelope = routing_envelope

    # Bounds
    if routing_envelope is not None and hasattr(routing_envelope, "bbox_nm"):
        g._bounds = tuple(int(v) for v in routing_envelope.bbox_nm)
    else:
        for plist in pins_by_net.values():
            for p in plist:
                px, py = _pin_xy_nm(p)
                all_rects_for_bounds.append((px - 500, py - 500, px + 500, py + 500))
        if all_rects_for_bounds:
            pad = 2000
            g._bounds = (
                min(r[0] for r in all_rects_for_bounds) - pad,
                min(r[1] for r in all_rects_for_bounds) - pad,
                max(r[2] for r in all_rects_for_bounds) + pad,
                max(r[3] for r in all_rects_for_bounds) + pad,
            )
        else:
            g._bounds = (0, 0, 10000, 10000)

    # DeckConnectStack for verify_static_terminal_escapability
    conductors = list(dict.fromkeys(tuple(k) for k in t.metal_keys.values()))
    connections = []
    for lo in range(n_layers - 1):
        vk = t.via_keys.get(lo)
        if vk is not None:
            connections.append(
                DeckConnection(
                    cut=tuple(vk),
                    lower=tuple(t.gi_to_metal_key(lo)),
                    upper=tuple(t.gi_to_metal_key(lo + 1)),
                )
            )
    g._deck_connect_stack = DeckConnectStack(
        conductors=tuple(conductors),
        connections=tuple(connections),
    )

    g._book = _Book(n_layers)
    g._active_half_width_overrides = {}
    g._mim_carve_bottom_blockers_by_gi = {}
    g._mim_forbidden_cut_touch_by_lo = {}
    g._mim_bottom_routing_clearance_by_gi = {}
    g._mim_top_via_marker_clearance_by_lo = {}
    g._active_group_route_local_closure = False
    g._active_group_route_local_interface = None
    g._negotiation_port_legalize = False
    g._active_static_metal_scans = None
    g._active_static_cut_scans = None
    g._active_static_obstacle_scans = None
    g._active_mirror_legality = None
    g._hard_rejections = {}
    g._hard_rejection_samples = {}
    g._hard_rejection_generation = 0

    all_gis = list(range(n_layers))
    layer_map = {
        c.net: [
            gi
            for gi in all_gis
            if t.resolve_layer_to_pdk(t.gi_to_metal_key(gi)) in c.layers
        ]
        for c in constraints
        if isinstance(c, LayerSet)
    }
    net_nodes: Dict[str, List[Tuple[int, int, int]]] = {}
    net_node_pins: Dict[str, List[Any]] = {}
    net_layers: Dict[str, List[int]] = {}
    for net, plist in pins_by_net.items():
        allowed = layer_map.get(net, all_gis) or all_gis
        net_layers[net] = list(allowed)
        nodes_l: List[Tuple[int, int, int]] = []
        pins_l: List[Any] = []
        for p in plist:
            px, py = _pin_xy_nm(p)
            gi = t.resolve_layer_to_gi(p.layer)
            if gi is None:
                gi = allowed[0]
            nodes_l.append((px, py, int(gi)))
            pins_l.append(p)
        net_nodes[net] = nodes_l
        net_node_pins[net] = pins_l

    return g, net_nodes, net_node_pins, net_layers


def _nearest_foreign_blockers(
    engine: Any,
    gi: int,
    net: str,
    query_rect: Rect,
) -> List[Rect]:
    out: List[Rect] = []
    idx = getattr(engine, "_obs_idx", {}).get(gi)
    if idx is not None:
        for owner, rect in idx.query(query_rect):
            if owner != net:
                out.append(_norm_rect(rect))
    try:
        wide_map = engine._wide_indexes(False)
        for _wspace, core, _win, _att in wide_map.get(gi, ()):
            for owner, rect in core.query(query_rect):
                if owner != net:
                    out.append(_norm_rect(rect))
    except Exception:
        pass
    return out


def _ray_is_clear(
    engine: Any,
    gi: int,
    net: str,
    ray_rect: Rect,
    half: int,
    sp: int,
    env_bbox: Optional[Rect],
) -> bool:
    rx0, ry0, rx1, ry1 = ray_rect
    if env_bbox is not None and (
        rx0 - half < env_bbox[0]
        or ry0 - half < env_bbox[1]
        or rx1 + half > env_bbox[2]
        or ry1 + half > env_bbox[3]
    ):
        return False
    obs_idx = getattr(engine, "_obs_idx", {}).get(gi)
    if obs_idx is not None:
        q = (rx0 - half - sp, ry0 - half - sp, rx1 + half + sp, ry1 + half + sp)
        for owner, _rect in obs_idx.query(q):
            if owner != net:
                return False
    wide_rows = engine._wide_indexes(False).get(gi, ())
    for wspace, core, _win, _att in wide_rows:
        wq = (
            rx0 - half - int(wspace),
            ry0 - half - int(wspace),
            rx1 + half + int(wspace),
            ry1 + half + int(wspace),
        )
        for owner, _rect in core.query(wq):
            if owner != net:
                return False
    return True


def _via_pad_clearance_for_redundancy(
    engine: Any,
    net: str,
    px: int,
    py: int,
    lo: int,
) -> Tuple[bool, Optional[Rect], int]:
    """Check whether any ``via_redundancy=N`` landing pad covering ``(px, py)`` clears foreign static obstacles.

    Returns ``(ok, tightest_blocker_bbox, min_expand_nm)``.
    """
    if lo < 0 or lo >= engine._max_gi:
        return False, None, 0
    bh, ah = engine._via_pad_half[lo]
    sp_lo = int(engine._space[lo])
    sp_hi = int(engine._space[lo + 1])
    grid = max(5, int(getattr(engine, "_manufacturing_grid_nm", 5)))

    # Probe centered + 4 cardinal corner offsets keeping pad over (px, py)
    step = max(20, bh // 3)
    offsets = sorted({0, -bh, bh, -step, step})
    best_blocker: Optional[Rect] = None
    smallest_deficit: Optional[int] = None

    for oy in offsets:
        for ox in offsets:
            cx, cy = px + ox, py + oy
            pad_lo = (cx - bh, cy - bh, cx + bh, cy + bh)
            pad_hi = (cx - ah, cy - ah, cx + ah, cy + ah)
            # Check static envelope bounds
            env = getattr(engine, "_routing_envelope", None)
            if env is not None and hasattr(env, "bbox_nm"):
                ex0, ey0, ex1, ey1 = env.bbox_nm
                if (
                    pad_lo[0] < ex0 or pad_lo[1] < ey0 or pad_lo[2] > ex1 or pad_lo[3] > ey1
                    or pad_hi[0] < ex0 or pad_hi[1] < ey0 or pad_hi[2] > ex1 or pad_hi[3] > ey1
                ):
                    continue

            q_lo = (pad_lo[0] - sp_lo, pad_lo[1] - sp_lo, pad_lo[2] + sp_lo, pad_lo[3] + sp_lo)
            q_hi = (pad_hi[0] - sp_hi, pad_hi[1] - sp_hi, pad_hi[2] + sp_hi, pad_hi[3] + sp_hi)
            block_lo = _nearest_foreign_blockers(engine, lo, net, q_lo)
            block_hi = _nearest_foreign_blockers(engine, lo + 1, net, q_hi)
            if not block_lo and not block_hi:
                return True, None, 0

            # Track minimum expansion needed at this offset site
            site_deficit = 0
            site_box: Optional[Rect] = None
            for gi_chk, pad, sp, blocks in (
                (lo, pad_lo, sp_lo, block_lo),
                (lo + 1, pad_hi, sp_hi, block_hi),
            ):
                del gi_chk
                for br in blocks:
                    dx, dy = _rect_gap_nm((px, py, px, py), br)
                    # Required clearance from foreign obstacle to pin centre
                    req = (pad[2] - pad[0]) // 2 + sp
                    axis_gap = min(dx, dy) if (dx > 0 and dy > 0) else max(dx, dy)
                    d = req - axis_gap
                    if d > site_deficit:
                        site_deficit = d
                        site_box = br
            if site_deficit <= 0:
                site_deficit = grid
            if smallest_deficit is None or site_deficit < smallest_deficit:
                smallest_deficit = site_deficit
                best_blocker = site_box

    return False, best_blocker, _ceil_grid(smallest_deficit or grid, grid)


def _compute_pin_escape_blocker(
    engine: Any,
    net: str,
    pin: Any,
    node: Tuple[int, int, int],
    allowed_gis: Sequence[int],
    via_redundancy: int,
) -> Optional[PreRouteBlocker]:
    """Check one pin for unexempted planar/via escape and return exact blocker if sealed."""
    from clay.layout.route.planar_precheck import _build_conservative_free_regions

    px, py, gi = node
    lg_set = set(allowed_gis)
    hw = int(engine._half_w[gi]) if gi < len(engine._half_w) else 115
    sp = int(engine._space[gi]) if gi < len(engine._space) else 230
    grid = max(5, int(getattr(engine, "_manufacturing_grid_nm", 5)))
    pin_id = f"{getattr(pin, 'device', '')}.{getattr(pin, 'terminal', '')}".strip(".") or f"{net}@{px},{py}"

    env = getattr(engine, "_routing_envelope", None)
    env_bbox = env.bbox_nm if env is not None and hasattr(env, "bbox_nm") else None

    # 1. If pin layer is allowed, test cardinal planar rays + conservative free region
    planar_rays = (
        (px, py, px + _ESCAPE_REACH_NM, py),
        (px - _ESCAPE_REACH_NM, py, px, py),
        (px, py, px, py + _ESCAPE_REACH_NM),
        (px, py - _ESCAPE_REACH_NM, px, py),
    )
    planar_ray_ok = (gi in lg_set) and any(
        _ray_is_clear(engine, gi, net, ray, hw, sp, env_bbox) for ray in planar_rays
    )

    via_ok = False
    via_lo_candidates = [lo for lo in (gi, gi - 1) if 0 <= lo < engine._max_gi and (lo in lg_set or (lo + 1) in lg_set)]
    tight_via_blocker: Optional[Rect] = None
    tight_via_expand: int = 0

    for lo in via_lo_candidates:
        ok, blk, exp_nm = _via_pad_clearance_for_redundancy(engine, net, px, py, lo)
        if ok:
            via_ok = True
            break
        if blk is not None and (tight_via_expand == 0 or exp_nm < tight_via_expand):
            tight_via_blocker = blk
            tight_via_expand = exp_nm

    if planar_ray_ok:
        # Planar escape on pin layer is wide open; check whether pin layer itself cannot reach
        # an upper preferred routing layer at the pin AND no planar ray leaves the tight region.
        return None
    if via_ok:
        return None

    # Conservative pocket check (+-1600nm) with owned-metal closure
    layers_local = sorted({gi} | ({gi + 1} & lg_set) | ({gi - 1} & lg_set))
    win = (px - _ESCAPE_REACH_NM, py - _ESCAPE_REACH_NM, px + _ESCAPE_REACH_NM, py + _ESCAPE_REACH_NM)
    F_wire, F_via = _build_conservative_free_regions(
        engine,
        net,
        layers_local,
        win,
        min_width_only=True,
        static_only=True,
    )
    seed = db.Region(db.Box(px, py, px + 1, py + 1))
    owned_reg = db.Region()
    for atom in getattr(engine, "_owned_terminal_atoms", ()):
        if str(atom.owner) == net and int(atom.gi) == gi:
            r = atom.rect
            owned_reg.insert(db.Box(r[0], r[1], r[2] + 1, r[3] + 1))
    owned_reg.merge()
    if not owned_reg.is_empty():
        conn = owned_reg.interacting(seed)
        if not conn.is_empty():
            seed = (seed + conn).merged()
    fw_gi = F_wire.get(gi, db.Region())
    reach_reg = (fw_gi.interacting(seed) + seed).merged()
    if not owned_reg.is_empty():
        for _ in range(32):
            joined = (reach_reg + owned_reg.interacting(reach_reg)).merged()
            expanded = (joined + fw_gi.interacting(joined)).merged()
            if (expanded - reach_reg).is_empty():
                break
            reach_reg = expanded

    bb = reach_reg.bbox()
    reaches_window = (gi in lg_set) and (
        bb.left <= px - _ESCAPE_REACH_NM + 10
        or bb.right >= px + _ESCAPE_REACH_NM - 10
        or bb.bottom <= py - _ESCAPE_REACH_NM + 10
        or bb.top >= py + _ESCAPE_REACH_NM - 10
    )
    if reaches_window:
        return None

    has_pocket_via = any(
        lo in F_via and not (F_via[lo] & reach_reg).is_empty()
        for lo in (gi, gi - 1)
    )
    if has_pocket_via:
        return None

    # Compute exact foreign blocker bbox & minimal corridor expansion in nm
    q_pocket = (px - _ESCAPE_REACH_NM, py - _ESCAPE_REACH_NM, px + _ESCAPE_REACH_NM, py + _ESCAPE_REACH_NM)
    foreign_blocks = _nearest_foreign_blockers(engine, gi, net, q_pocket)
    pin_pt = (px, py, px, py)
    nearest_blk: Optional[Rect] = tight_via_blocker
    min_exp = tight_via_expand
    for br in foreign_blocks:
        dx, dy = _rect_gap_nm(pin_pt, br)
        gap = max(dx, dy) if (dx == 0 or dy == 0) else min(dx, dy)
        raw_deficit = (hw + sp) - gap
        if raw_deficit <= 0:
            continue
        deficit = _ceil_grid(raw_deficit, grid)
        if min_exp == 0 or deficit < min_exp:
            min_exp = deficit
            nearest_blk = br
    if nearest_blk is None and foreign_blocks:
        nearest_blk = _union_bbox_nm(foreign_blocks)
    if min_exp <= 0:
        min_exp = _ceil_grid(2 * hw + sp, grid)

    is_via_squeeze = tight_via_blocker is not None and via_redundancy > 1
    cat = "via_landing_clearance" if is_via_squeeze else "pin_escape"
    return PreRouteBlocker(
        tier="T2",
        category=cat,
        net=net,
        pin=pin_id,
        blocker_bbox_nm=nearest_blk,
        min_expand_nm=min_exp,
        reason=(
            f"{'VIA_REDUNDANCY_PAD_BLOCKED' if is_via_squeeze else 'PIN_ESCAPE_SEALED'}: "
            f"pin {pin_id} on net {net!r} (M{gi + 1} @ {px},{py}nm) has no planar escape "
            f"and no legal via_redundancy={via_redundancy} landing"
        ),
        suggested_action=(
            f"Expand local placement corridor around {pin_id} ({net}) by >= "
            f"{min_exp}nm ({min_exp / 1000.0:.3f}um) to restore pin escape / via_redundancy={via_redundancy} clearance"
        ),
        detail={
            "pin_xy_nm": [px, py],
            "layer_gi": gi,
            "via_redundancy": via_redundancy,
            "reaches_window": False,
        },
    )


def _overlap_compatible_pairs_and_owners(
    final_placement_certificate: Any,
) -> Tuple[Set[Tuple[str, str]], Dict[str, str]]:
    """Extract certified layer-disjoint master pairs and member->master ownership."""
    overlap_pairs: Set[Tuple[str, str]] = set()
    owner_of: Dict[str, str] = {}
    if final_placement_certificate is None:
        return overlap_pairs, owner_of
    raw_pairs = getattr(final_placement_certificate, "overlap_compatible_pairs", ()) or ()
    for pair in raw_pairs:
        if isinstance(pair, (tuple, list)) and len(pair) == 2:
            a_m, b_m = str(pair[0]), str(pair[1])
            if a_m and b_m:
                overlap_pairs.add((a_m, b_m) if a_m <= b_m else (b_m, a_m))
    for mapping_attr in (
        "materialized_block_source_members",
        "materialized_block_members",
    ):
        rows = getattr(final_placement_certificate, mapping_attr, ()) or ()
        for master, members in rows:
            master_s = str(master)
            for member in members:
                owner_of.setdefault(str(member), master_s)
    return overlap_pairs, owner_of


def _check_t2_corridor_congestion(
    *,
    engine: Any,
    tech: Any,
    dev_bboxes: Mapping[str, Rect],
    pins_by_net: Mapping[str, Sequence[Any]],
    constraints: Sequence[Any],
    final_placement_certificate: Any = None,
) -> Tuple[List[PreRouteBlocker], Dict[str, Any]]:
    """Estimate H/V inter-device corridor track supply vs crossing-net demand."""
    from clay.layout.route.constraints import Shield, Spacing, Width
    from clay.layout.route.gridless_types import _pin_xy_nm
    from clay.layout.route.lower import conventional_axes_for

    if len(dev_bboxes) < 2:
        return [], {"corridors_checked": 0, "worst_utilization": 0.0}

    overlap_pairs, owner_of = _overlap_compatible_pairs_and_owners(
        final_placement_certificate
    )
    obs_idx = getattr(engine, "_obs_idx", None) or {}

    def _is_overlap_compatible(da: str, db_name: str) -> bool:
        if not overlap_pairs:
            return False
        ma = owner_of.get(da)
        mb = owner_of.get(db_name)
        if not ma or not mb or ma == mb:
            return False
        return ((ma, mb) if ma <= mb else (mb, ma)) in overlap_pairs

    axes = conventional_axes_for(tech, n_max=max(4, engine._n_layers))
    grid = max(5, int(getattr(engine, "_manufacturing_grid_nm", 5)))
    layer_pitches: Dict[str, List[int]] = {"H": [], "V": []}
    axis_layers: Dict[str, List[Tuple[int, int, int]]] = {"H": [], "V": []}
    for gi in range(engine._n_layers):
        pdk_idx = engine._tech._gi_to_pdk[gi]
        ax = axes.get(pdk_idx, "H" if pdk_idx % 2 == 1 else "V")
        gp = max(1, int(engine._tech.min_width_nm[pdk_idx] + engine._tech.min_space_nm[pdk_idx]))
        gsp = max(0, int(engine._tech.min_space_nm[pdk_idx]))
        layer_pitches[ax].append(gp)
        axis_layers[ax].append((gi, gp, gsp))
    pitch_h = min(layer_pitches["H"]) if layer_pitches["H"] else 560
    pitch_v = min(layer_pitches["V"]) if layer_pitches["V"] else 560

    def _unobstructed_over_cell_tracks(axis: str, qbox: Rect, span_nm: int) -> int:
        if span_nm <= 0:
            return 0
        qx0, qy0, qx1, qy1 = qbox
        if qx1 <= qx0 or qy1 <= qy0:
            return 0
        total = 0
        for gi, gp, gsp in axis_layers[axis]:
            idx = obs_idx.get(gi)
            obstructed = False
            if idx is not None:
                for _own, (rx0, ry0, rx1, ry1) in idx.query(qbox):
                    if rx1 > qx0 and rx0 < qx1 and ry1 > qy0 and ry0 < qy1:
                        obstructed = True
                        break
            if not obstructed:
                total += max(0, (span_nm - gsp) // gp)
        return total

    # Extra track weights from constraints
    net_track_weight: Dict[str, float] = {}
    for net in pins_by_net:
        net_track_weight[net] = 1.0
    for c in constraints:
        if isinstance(c, Width) and c.net in net_track_weight:
            w_nm = float(c.width_um) * _NM_PER_UM
            if w_nm > 0:
                net_track_weight[c.net] = max(net_track_weight[c.net], max(1.0, w_nm / max(1, min(pitch_h, pitch_v))))
        elif isinstance(c, Shield):
            net_track_weight[c.victim] = net_track_weight.get(c.victim, 1.0) + 1.0
        elif isinstance(c, Spacing):
            base_sp = min(engine._space) if engine._space else 230
            extra = max(0.0, (float(c.min_um) * _NM_PER_UM - base_sp) / max(1, min(pitch_h, pitch_v)))
            if extra > 0:
                net_track_weight[c.net_a] = net_track_weight.get(c.net_a, 1.0) + 0.5 * extra
                net_track_weight[c.net_b] = net_track_weight.get(c.net_b, 1.0) + 0.5 * extra

    net_spans: Dict[str, Rect] = {}
    for net, plist in pins_by_net.items():
        if len(plist) < 2 and net not in engine._rail_by_net:
            continue
        pts = [_pin_xy_nm(p) for p in plist]
        if pts:
            net_spans[net] = (
                min(x for x, _ in pts),
                min(y for _, y in pts),
                max(x for x, _ in pts),
                max(y for _, y in pts),
            )

    items = sorted(dev_bboxes.items())
    blockers: List[PreRouteBlocker] = []
    worst_util = 0.0
    corridors_checked = 0

    for i, (da, ba) in enumerate(items):
        ax0, ay0, ax1, ay1 = ba
        for db_name, bb in items[i + 1 :]:
            bx0, by0, bx1, by1 = bb
            # Horizontal corridor (gap in Y, overlapping X extent)
            ov_x0, ov_x1 = max(ax0, bx0), min(ax1, bx1)
            gap_y = max(ay0, by0) - min(ay1, by1)
            if ov_x1 > ov_x0 and gap_y > 0:
                corridors_checked += 1
                cy0, cy1 = min(ay1, by1), max(ay0, by0)
                cbox = (ov_x0, cy0, ov_x1, cy1)
                mid_y = (cy0 + cy1) // 2
                # Count nets crossing this Y cutline inside/near X span
                crossing = [
                    net
                    for net, (nx0, ny0, nx1, ny1) in net_spans.items()
                    if ny0 <= cy1 and ny1 >= cy0 and nx1 >= ov_x0 - 500 and nx0 <= ov_x1 + 500
                    and (ny0 < mid_y < ny1)
                ]
                demand = sum(net_track_weight.get(n, 1.0) for n in crossing)
                h_layers = max(1, len(layer_pitches["H"]))
                supply = max(0, ((gap_y - min(engine._space or [230])) // max(1, pitch_h))) * h_layers
                if _is_overlap_compatible(da, db_name):
                    qbox = (ov_x0, min(ay0, by0), ov_x1, max(ay1, by1))
                    supply += _unobstructed_over_cell_tracks("H", qbox, max(ay1, by1) - min(ay0, by0))
                util = demand / max(0.5, float(supply)) if demand > 0 else 0.0
                worst_util = max(worst_util, util)
                if demand > supply and crossing:
                    deficit = _ceil_grid(((demand - supply) / h_layers) * pitch_h, grid)
                    blockers.append(
                        PreRouteBlocker(
                            tier="T2",
                            category="corridor_congestion",
                            net=",".join(sorted(crossing)[:4]),
                            pin=f"corridor:{da}<->{db_name}",
                            blocker_bbox_nm=cbox,
                            min_expand_nm=max(grid, deficit),
                            reason=(
                                f"CORRIDOR_CONGESTION_H: channel {da}<->{db_name} "
                                f"gap_y={gap_y}nm supplies ~{supply} H-tracks < "
                                f"demand {demand:.1f} across {len(crossing)} nets"
                            ),
                            suggested_action=(
                                f"Expand vertical row gap between {da} and {db_name} by >= "
                                f"{max(grid, deficit)}nm ({max(grid, deficit) / 1000.0:.3f}um)"
                            ),
                            detail={
                                "axis": "H",
                                "gap_nm": gap_y,
                                "supply_tracks": supply,
                                "demand_tracks": round(demand, 2),
                                "crossing_nets": sorted(crossing),
                            },
                        )
                    )

            # Vertical corridor (gap in X, overlapping Y extent)
            ov_y0, ov_y1 = max(ay0, by0), min(ay1, by1)
            gap_x = max(ax0, bx0) - min(ax1, bx1)
            if ov_y1 > ov_y0 and gap_x > 0:
                corridors_checked += 1
                cx0, cx1 = min(ax1, bx1), max(ax0, bx0)
                cbox = (cx0, ov_y0, cx1, ov_y1)
                mid_x = (cx0 + cx1) // 2
                crossing = [
                    net
                    for net, (nx0, ny0, nx1, ny1) in net_spans.items()
                    if nx0 <= cx1 and nx1 >= cx0 and ny1 >= ov_y0 - 500 and ny0 <= ov_y1 + 500
                    and (nx0 < mid_x < nx1)
                ]
                demand = sum(net_track_weight.get(n, 1.0) for n in crossing)
                v_layers = max(1, len(layer_pitches["V"]))
                supply = max(0, ((gap_x - min(engine._space or [230])) // max(1, pitch_v))) * v_layers
                if _is_overlap_compatible(da, db_name):
                    qbox = (min(ax0, bx0), ov_y0, max(ax1, bx1), ov_y1)
                    supply += _unobstructed_over_cell_tracks("V", qbox, max(ax1, bx1) - min(ax0, bx0))
                util = demand / max(0.5, float(supply)) if demand > 0 else 0.0
                worst_util = max(worst_util, util)
                if demand > supply and crossing:
                    deficit = _ceil_grid(((demand - supply) / v_layers) * pitch_v, grid)
                    blockers.append(
                        PreRouteBlocker(
                            tier="T2",
                            category="corridor_congestion",
                            net=",".join(sorted(crossing)[:4]),
                            pin=f"corridor:{da}<->{db_name}",
                            blocker_bbox_nm=cbox,
                            min_expand_nm=max(grid, deficit),
                            reason=(
                                f"CORRIDOR_CONGESTION_V: channel {da}<->{db_name} "
                                f"gap_x={gap_x}nm supplies ~{supply} V-tracks < "
                                f"demand {demand:.1f} across {len(crossing)} nets"
                            ),
                            suggested_action=(
                                f"Expand horizontal column gap between {da} and {db_name} by >= "
                                f"{max(grid, deficit)}nm ({max(grid, deficit) / 1000.0:.3f}um)"
                            ),
                            detail={
                                "axis": "V",
                                "gap_nm": gap_x,
                                "supply_tracks": supply,
                                "demand_tracks": round(demand, 2),
                                "crossing_nets": sorted(crossing),
                            },
                        )
                    )

    return blockers, {
        "corridors_checked": corridors_checked,
        "worst_utilization": round(worst_util, 3),
    }


def _bridge_router_inputs(
    placement: Any,
    cell_def: Any,
    intent: Any,
    tech: Any,
    *,
    device_records: Optional[Mapping[str, Mapping[str, Any]]],
    effective_policy: Any,
    dense_contacts: bool = False,
    placement_um_override: Optional[Mapping[str, Tuple[float, float]]] = None,
    geometry_override: Optional[Mapping[str, Any]] = None,
    final_placement_certificate: Any = None,
    member_ring_pad_top_nm: Sequence[Tuple[str, int]] = (),
    member_ring_pad_bottom_nm: Sequence[Tuple[str, int]] = (),
    shared_well_guard_envelope_plans: Sequence[Any] = (),
    shared_well_guard_ring_inventory: Any = None,
    engine: Any = None,
    route_power: bool = False,
) -> Tuple[Dict[str, List[Any]], List[Any], Dict[str, Any], Dict[str, Tuple[float, float]], List[Any], Any]:
    """Bridge ``(placement, cell_def, intent)`` into pins, obstacles, constraints, envelope."""
    from dataclasses import replace as _dc_replace

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.model import build_layout_model
    from clay.layout.route.lower import conventional_axes_for, lower_intent
    from clay.layout.route.pins import extract_pins
    from clay.layout.route.rc_bridge import (
        _bridge_placement_to_route_inputs,
        _declared_ground_ports,
        _extract_native_row_pin_report,
        _relocate_array_member_pins,
    )
    from clay.layout.route.rc_intent_constraints import _routing_intent_for
    from clay.layout.route.rc_rails_guards import (
        _effective_guard_ring_intents,
        _materialize_guard_rings,
        _routing_envelope_from_final_placement,
        _routing_envelope_from_geometry,
        _validate_guard_ring_member_padding,
    )

    eff_intent = intent
    if eff_intent is None and cell_def is not None:
        eff_intent = layout_intent_from_cell_def(cell_def)

    guard_ring_intents = (
        _effective_guard_ring_intents(
            eff_intent,
            shared_well_guard_envelope_plans=shared_well_guard_envelope_plans,
            shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
        )
        if eff_intent is not None
        else ()
    )
    if eff_intent is not None and tuple(guard_ring_intents) != tuple(getattr(eff_intent, "guard_rings", ())):
        eff_intent = _dc_replace(eff_intent, guard_rings=guard_ring_intents)

    bridged_intent, placement_um, geometry = _bridge_placement_to_route_inputs(
        placement,
        cell_def,
        tech.pdk_id,
        device_records=device_records,
        dense_contacts=dense_contacts,
        intent=eff_intent,
    )
    has_override = placement_um_override is not None and geometry_override is not None
    final_row_placement = placement
    if has_override:
        drop_names = {
            n
            for n, cg in (geometry_override or {}).items()
            if str(getattr(cg, "source", "")) in ("group_guard_ring", "power_rail", "power_plane")
            or str(n).startswith("GRING")
            or str(n).startswith("PRAIL_")
            or str(n).startswith("PPLANE_")
        }
        placement_um = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in dict(placement_um_override).items()
            if str(k) not in drop_names
        }
        geometry_copy = {
            k: v for k, v in dict(geometry_override).items() if str(k) not in drop_names
        }
    else:
        geometry_copy = dict(geometry)

    if guard_ring_intents:
        try:
            if has_override and final_placement_certificate is not None:
                ring_pad_rows = _validate_guard_ring_member_padding(
                    member_ring_pad_top_nm,
                    placement_override=(placement_um, geometry_copy),
                    final_placement_certificate=final_placement_certificate,
                )
                ring_bottom_pad_rows = _validate_guard_ring_member_padding(
                    member_ring_pad_bottom_nm,
                    placement_override=(placement_um, geometry_copy),
                    final_placement_certificate=final_placement_certificate,
                )
            else:
                ring_pad_rows = dict(member_ring_pad_top_nm)
                ring_bottom_pad_rows = dict(member_ring_pad_bottom_nm)
            _materialize_guard_rings(
                guard_ring_intents,
                bridged_intent,
                placement_um,
                geometry_copy,
                tech,
                tech.pdk_id,
                member_ring_pad_top_nm=dict(ring_pad_rows),
                member_ring_pad_bottom_nm=dict(ring_bottom_pad_rows),
            )
        except Exception:
            pass

    if has_override and placement is not None and eff_intent is not None:
        from clay.layout.flow_placement import project_final_route_placement

        try:
            final_row_placement = project_final_route_placement(
                placement,
                placement_um,
                geometry_copy,
                final_placement_certificate=final_placement_certificate,
                intent=eff_intent,
            )
        except Exception:
            final_row_placement = placement

    routing_intent = _routing_intent_for(eff_intent, bridged_intent, tech)
    extra_power_nets = _declared_ground_ports(cell_def) if cell_def is not None else ()
    if getattr(final_row_placement, "native_geometry", None) is not None:
        pin_report = _extract_native_row_pin_report(
            bridged_intent,
            placement_um,
            tech=tech,
            geometry=geometry_copy,
            intent=eff_intent,
            source_cell=cell_def,
            placement_obj=final_row_placement,
            route_engine=engine,
            include_power=True,
            extra_power_nets=extra_power_nets,
        )
    else:
        pin_report = extract_pins(
            bridged_intent,
            placement_um,
            tech=tech,
            geometry=geometry_copy,
            return_report=True,
            include_power=bool(route_power),
            per_device_fold=True,
            extra_power_nets=extra_power_nets,
        )
    pins_by_net = {str(k): list(v) for k, v in (pin_report.pins_by_net or {}).items()}
    _relocate_array_member_pins(
        pins_by_net,
        bridged_intent,
        cell_def,
        eff_intent,
        tech,
        placement_um,
        device_records=device_records,
        placement_obj=final_row_placement,
        geometry=geometry_copy,
        allow_single_component_joint=bool(
            getattr(engine, "consumes_single_component_joint_terminals", False)
        ),
        allow_fixed_multilayer_joint=bool(
            getattr(engine, "consumes_fixed_multilayer_joint_terminals", False)
        ),
        multilayer_engine=engine,
    )
    model = build_layout_model(
        bridged_intent,
        placement_um,
        geometry_copy,
        tech=tech,
        pins_by_net=pins_by_net,
        include_upper_artwork=True,
        strict_device_m1_provenance=bool(
            getattr(engine, "strict_device_m1_provenance", False)
        ),
        include_device_projection=False,
    )
    obstacles = list(model.router_obstacles())
    axes = conventional_axes_for(tech)
    constraints = lower_intent(routing_intent, axes=axes) if routing_intent is not None else []
    routing_envelope = (
        _routing_envelope_from_geometry(placement_um, geometry_copy)
        if geometry_copy
        else None
    )
    if routing_envelope is not None and final_placement_certificate is not None:
        routing_envelope = _routing_envelope_from_final_placement(
            routing_envelope,
            final_placement_certificate,
        )
    if getattr(final_row_placement, "ordered_row_evidence", None) is not None and eff_intent is not None:
        from clay.layout.place.ordered_mirror_rows import verify_ordered_row_floorplan
        from clay.layout.route.floorplan_containment import RoutingEnvelope

        try:
            ordered_floorplan_nm = verify_ordered_row_floorplan(
                final_row_placement,
                eff_intent,
                tech,
                geometry=geometry_copy,
                placement_um=placement_um,
            )
            routing_envelope = RoutingEnvelope.coerce(ordered_floorplan_nm)
        except Exception:
            pass
    return pins_by_net, obstacles, geometry_copy, placement_um, constraints, routing_envelope


def _check_t2_shield_separability(
    *,
    cell_def: Any,
    intent: Any,
    tech: Any,
    pins_by_net: Mapping[str, Sequence[Any]],
    geometry: Optional[Mapping[str, Any]],
    placement_um: Optional[Mapping[str, Tuple[float, float]]],
    mos_devices: Sequence[Any],
    passives: Sequence[Any],
    dev_bboxes: Mapping[str, Tuple[int, int, int, int]],
    grid_nm: int = 5,
) -> List[PreRouteBlocker]:
    """Check T2 terminal-domain separability for declared @shield fences."""
    from clay.layout.intent import layout_intent_from_cell_def, replay_shield_inventory
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.shield import (
        ShieldMaterializationError,
        installed_shield_layer_policy,
        materialize_declared_shield_fences,
    )

    try:
        grid_nm = int(pdk_manufacturing_grid_nm(tech))
    except Exception:
        grid_nm = int(grid_nm or 5)

    eff_intent = intent
    if eff_intent is None and cell_def is not None:
        try:
            eff_intent = layout_intent_from_cell_def(cell_def)
        except Exception:
            eff_intent = None
    if eff_intent is None:
        return []
    raw_shields = tuple(getattr(eff_intent, "shields", ()) or ())
    if not raw_shields:
        return []

    fallback_bbox = _union_bbox_nm(list(dev_bboxes.values()))
    try:
        typed = replay_shield_inventory(raw_shields)
    except (TypeError, ValueError) as exc:
        return [
            PreRouteBlocker(
                tier="T2",
                category="shield_domain_nonseparable",
                net="",
                pin="",
                blocker_bbox_nm=fallback_bbox,
                min_expand_nm=_ceil_grid(560, max(1, int(grid_nm))),
                reason=f"SHIELD_INVENTORY_INVALID: {exc}",
                suggested_action="Inspect @shield(agg -> vic, shield) net and domain syntax",
            )
        ]
    if not typed:
        return []

    try:
        layer_by_axis, shield_grid_nm, minimum_width_nm = installed_shield_layer_policy(tech)
    except ShieldMaterializationError as exc:
        return [
            PreRouteBlocker(
                tier="T2",
                category="shield_domain_nonseparable",
                net=f"{typed[0].aggressor}->{typed[0].victim}",
                pin=str(typed[0].victim),
                blocker_bbox_nm=fallback_bbox,
                min_expand_nm=_ceil_grid(560, max(1, int(grid_nm))),
                reason=f"SHIELD_LAYER_POLICY_FAILED: {exc}",
                suggested_action="Verify PDK installs H/V transport metal rules required by @shield",
            )
        ]

    owners = {
        str(getattr(d, "name", "") or "")
        for d in (*mos_devices, *passives)
        if getattr(d, "name", None)
    }
    owners.update(str(k) for k in dev_bboxes)
    if cell_def is not None:
        for d in getattr(cell_def, "devices", ()) or ():
            iname = str(getattr(d, "instance_name", "") or "")
            if iname:
                owners.add(iname)
    domain_pin_owners = frozenset(owners) if owners else None

    known_nets: Dict[str, str] = {}
    for net in pins_by_net:
        s = str(net)
        known_nets.setdefault(s.rstrip("!").casefold(), s)
    if cell_def is not None:
        for port in getattr(cell_def, "ports", ()) or ():
            s = str(port)
            if s:
                known_nets.setdefault(s.rstrip("!").casefold(), s)
        for d in getattr(cell_def, "devices", ()) or ():
            pins_map = getattr(d, "pins", None) or {}
            if isinstance(pins_map, Mapping):
                for n in pins_map.values():
                    s = str(n or "")
                    if s:
                        known_nets.setdefault(s.rstrip("!").casefold(), s)

    has_geom = bool(geometry) and bool(placement_um)
    wire_widths_raw = getattr(eff_intent, "wire_widths", None)
    wire_widths_nm = dict(wire_widths_raw) if isinstance(wire_widths_raw, Mapping) and wire_widths_raw else None
    step_grid = max(1, int(shield_grid_nm or grid_nm or 5))
    min_exp = _ceil_grid(max(2 * int(minimum_width_nm), 280), step_grid)

    def _domain_bbox(agg: str, vic: str) -> Tuple[int, int, int, int]:
        pts: List[Tuple[int, int]] = []
        agg_norm = agg.rstrip("!").casefold()
        vic_norm = vic.rstrip("!").casefold()
        for net, plist in pins_by_net.items():
            if str(net).rstrip("!").casefold() not in (agg_norm, vic_norm):
                continue
            for p in plist:
                if domain_pin_owners is None or getattr(p, "device", None) in domain_pin_owners:
                    xy_nm = getattr(p, "xy_nm", None)
                    if xy_nm is not None and len(xy_nm) == 2:
                        pts.append((int(xy_nm[0]), int(xy_nm[1])))
                    else:
                        xy_um = getattr(p, "xy_um", None)
                        if xy_um is not None and len(xy_um) == 2:
                            pts.append((round(float(xy_um[0]) * 1000.0), round(float(xy_um[1]) * 1000.0)))
        if not pts:
            return fallback_bbox
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pad = max(step_grid, int(minimum_width_nm) // 2)
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    blockers: List[PreRouteBlocker] = []
    for decl in typed:
        pins_copy: Dict[str, List[Any]] = {str(k): list(v) for k, v in pins_by_net.items()}
        shield_norm = decl.shield_net.rstrip("!").casefold()
        if not any(str(k).rstrip("!").casefold() == shield_norm for k in pins_copy):
            if shield_norm in known_nets:
                pins_copy[known_nets[shield_norm]] = []
        try:
            materialize_declared_shield_fences(
                (decl,),
                pins_copy,
                layer_by_axis=layer_by_axis,
                manufacturing_grid_nm=shield_grid_nm,
                minimum_width_nm=minimum_width_nm,
                domain_pin_owners=domain_pin_owners,
                emitted_device_geometry=geometry if has_geom else None,
                emitted_device_placement_um=placement_um if has_geom else None,
                tech=tech if has_geom else None,
                wire_widths_nm=wire_widths_nm,
            )
        except ShieldMaterializationError as exc:
            blockers.append(
                PreRouteBlocker(
                    tier="T2",
                    category="shield_domain_nonseparable",
                    net=f"{decl.aggressor}->{decl.victim}",
                    pin=str(decl.victim),
                    blocker_bbox_nm=_domain_bbox(decl.aggressor, decl.victim),
                    min_expand_nm=min_exp,
                    reason=f"SHIELD_DOMAIN_NONSEPARABLE: {exc}",
                    suggested_action=(
                        f"Remove shared-device @shield({decl.aggressor} -> {decl.victim}, {decl.shield_net}) "
                        "or separate the aggressor and victim device islands across rows/columns "
                        f"by >= {min_exp}nm to open a legal shield fence corridor"
                    ),
                    detail={
                        "aggressor": decl.aggressor,
                        "victim": decl.victim,
                        "shield_net": decl.shield_net,
                    },
                )
            )
    return blockers


def run_preroute_gate(
    placement: Any = None,
    cell_def: Any = None,
    intent: Any = None,
    pdk: Any = "gf180mcu",
    *,
    device_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    effective_policy: Any = None,
    pins_by_net: Optional[Mapping[str, Sequence[Any]]] = None,
    obstacles: Optional[Sequence[Any]] = None,
    geometry: Optional[Mapping[str, Any]] = None,
    placement_um: Optional[Mapping[str, Tuple[float, float]]] = None,
    constraints: Optional[Sequence[Any]] = None,
    routing_envelope: Any = None,
    engine: Any = None,
    group_route_contract: Any = None,
    via_redundancy: Optional[int] = None,
    pex_backend: Optional[str] = None,
    cap_option: Optional[str] = None,
    mim_option: Optional[str] = None,
    physical: bool = False,
    source_artifact: Optional[str] = None,
    final_placement_certificate: Any = None,
    member_ring_pad_top_nm: Sequence[Tuple[str, int]] = (),
    member_ring_pad_bottom_nm: Sequence[Tuple[str, int]] = (),
    shared_well_guard_envelope_plans: Sequence[Any] = (),
    shared_well_guard_ring_inventory: Any = None,
    route_power: bool = False,
    budget_sec: float = 5.0,
) -> PreRouteGateResult:
    """Execute static T1 (Source/PDK/LVS) and T2 (Pin-escape/Via/Corridor) gates in <5s.

    Does NOT skip ``verify_static_terminal_escapability`` when ``group_route_contract``
    is provided, and returns exact actionable ``PreRouteBlocker`` records whenever any
    check fails.
    """
    t0 = time.monotonic()
    deadline = t0 + max(0.25, float(budget_sec))

    pdk_id = _resolve_pdk_id(pdk, cell_def)
    tech = _resolve_tech(pdk, pdk_id)
    dev_bboxes = _device_bbox_map_nm(placement)
    if placement_um is not None and geometry is not None:
        try:
            from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm

            stamp_boxes = _placed_stamp_bboxes_nm(geometry, dict(placement_um))
            for k, box in stamp_boxes.items():
                dev_bboxes[str(k)] = _norm_rect(box)
        except Exception:
            pass
    pex_norm, cap_opt_norm = _resolve_pex_and_cap_option(
        pex_backend=pex_backend,
        cap_option=cap_option if cap_option is not None else mim_option,
        cell_def=cell_def,
        effective_policy=effective_policy,
        intent=intent,
    )

    # Resolve via_redundancy from explicit arg, engine, or effective_policy
    eff_vr = via_redundancy
    if eff_vr is None and engine is not None:
        eff_vr = getattr(engine, "via_redundancy", None)
    if eff_vr is None and effective_policy is not None:
        eff_vr = getattr(effective_policy, "via_redundancy", None)
    vr = max(1, int(eff_vr or 1))

    # --- T1: Source-PDK-LVS Pre-check (<0.5s) ---
    t1_blockers: List[PreRouteBlocker] = []
    mos_devices, passives = _extract_device_entries(cell_def, device_records)

    # Bridge placement->router world when raw pins/obstacles aren't supplied
    resolved_pins = {str(k): list(v) for k, v in pins_by_net.items()} if pins_by_net is not None else None
    resolved_obs = list(obstacles) if obstacles is not None else None
    resolved_geom = dict(geometry) if geometry is not None else None
    resolved_place_um = dict(placement_um) if placement_um is not None else None
    resolved_cons = list(constraints) if constraints is not None else []
    resolved_env = routing_envelope

    if resolved_pins is None and placement is not None and cell_def is not None:
        try:
            (
                resolved_pins,
                resolved_obs_b,
                geom_b,
                place_um_b,
                cons_b,
                env_b,
            ) = _bridge_router_inputs(
                placement,
                cell_def,
                intent,
                tech,
                device_records=device_records,
                effective_policy=effective_policy,
                placement_um_override=placement_um,
                geometry_override=geometry,
                final_placement_certificate=final_placement_certificate,
                member_ring_pad_top_nm=member_ring_pad_top_nm,
                member_ring_pad_bottom_nm=member_ring_pad_bottom_nm,
                shared_well_guard_envelope_plans=shared_well_guard_envelope_plans,
                shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
                engine=engine,
                route_power=route_power,
            )
            if resolved_obs is None:
                resolved_obs = resolved_obs_b
            if resolved_geom is None:
                resolved_geom = geom_b
            if resolved_place_um is None:
                resolved_place_um = place_um_b
            if not resolved_cons:
                resolved_cons = cons_b
            if resolved_env is None:
                resolved_env = env_b
        except Exception as exc:
            t1_blockers.append(
                PreRouteBlocker(
                    tier="T1",
                    category="bridge_failure",
                    net="",
                    pin="",
                    blocker_bbox_nm=_union_bbox_nm(list(dev_bboxes.values())),
                    min_expand_nm=0,
                    reason=f"PLACEMENT_BRIDGE_FAILED: {type(exc).__name__}: {exc}",
                    suggested_action="Inspect placement stamps and device pin bindings before routing",
                )
            )

    t1_blockers.extend(
        _check_t1_device_support(
            cell_def,
            pdk_id,
            physical=physical,
            source_artifact=source_artifact,
            dev_bboxes=dev_bboxes,
        )
    )
    t1_blockers.extend(
        _check_t1_gf180_mim_stack(
            pdk_id=pdk_id,
            tech=tech,
            passives=passives,
            geometry=resolved_geom,
            placement_um=resolved_place_um,
            dev_bboxes=dev_bboxes,
            pex_backend=pex_norm,
            cap_option=cap_opt_norm,
        )
    )
    t1_blockers.extend(
        _check_t1_well_and_substrate_closure(
            cell_def=cell_def,
            intent=intent,
            pdk_id=pdk_id,
            tech=tech,
            mos_devices=mos_devices,
            dev_bboxes=dev_bboxes,
        )
    )

    # --- T2: Unexempted Pin Escape, Via-Redundancy & Corridor Congestion ---
    t2_blockers: List[PreRouteBlocker] = []
    timed_out = False
    corridor_metrics: Dict[str, Any] = {"corridors_checked": 0, "worst_utilization": 0.0}

    if resolved_pins is not None and resolved_obs is not None:
        from clay.layout.route.planar_precheck import verify_static_terminal_escapability

        rem_budget = max(0.25, deadline - time.monotonic())
        g_engine, net_nodes, net_node_pins, net_layers = _populate_gridless_static_world(
            tech=tech,
            pins_by_net=resolved_pins,
            obstacles=resolved_obs,
            constraints=resolved_cons,
            routing_envelope=resolved_env,
            via_redundancy=vr,
            budget_sec=rem_budget,
            engine=engine,
        )

        # Unconditionally run verify_static_terminal_escapability even if
        # group_route_contract is present (zero silent exemptions).
        static_escape_exc: Optional[str] = None
        try:
            verify_static_terminal_escapability(
                g_engine,
                resolved_pins,
                net_nodes,
                net_node_pins,
                net_layers,
            )
        except ValueError as exc:
            static_escape_exc = str(exc)
        except Exception as exc:
            static_escape_exc = f"{type(exc).__name__}: {exc}"

        recorded_pin_keys: Set[Tuple[str, str]] = set()
        for net, nodes in net_nodes.items():
            if time.monotonic() >= deadline:
                timed_out = True
                break
            npins = net_node_pins.get(net, ())
            rails = getattr(g_engine, "_rail_by_net", {}).get(net, ())
            if len(nodes) < 2 and not rails:
                continue
            allowed_gis = net_layers.get(net, list(range(g_engine._n_layers)))
            for pin, node in zip(npins, nodes):
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                blk = _compute_pin_escape_blocker(
                    g_engine, net, pin, node, allowed_gis, vr
                )
                if blk is not None:
                    t2_blockers.append(blk)
                    recorded_pin_keys.add((blk.net, blk.pin))

        if static_escape_exc and not t2_blockers:
            # Ensure verify_static_terminal_escapability rejection is never swallowed
            t2_blockers.append(
                PreRouteBlocker(
                    tier="T2",
                    category="terminal_binding_escape",
                    net="",
                    pin="",
                    blocker_bbox_nm=_union_bbox_nm(list(dev_bboxes.values())),
                    min_expand_nm=_ceil_grid(
                        max(g_engine._half_w[0] + g_engine._space[0], 350),
                        g_engine._manufacturing_grid_nm,
                    ),
                    reason=f"STATIC_TERMINAL_ESCAPE_FAILED: {static_escape_exc}",
                    suggested_action=(
                        "Expand placement corridor around the reported isolated terminal "
                        "to open planar M1 exit or legal via landing"
                    ),
                    detail={"group_route_contract_present": group_route_contract is not None},
                )
            )

        if not timed_out:
            c_blockers, corridor_metrics = _check_t2_corridor_congestion(
                engine=g_engine,
                tech=tech,
                dev_bboxes=dev_bboxes,
                pins_by_net=resolved_pins,
                constraints=resolved_cons,
                final_placement_certificate=final_placement_certificate,
            )
            t2_blockers.extend(c_blockers)

    if resolved_pins is not None and not timed_out:
        shield_blockers = _check_t2_shield_separability(
            cell_def=cell_def,
            intent=intent,
            tech=tech,
            pins_by_net=resolved_pins,
            geometry=resolved_geom,
            placement_um=resolved_place_um,
            mos_devices=mos_devices,
            passives=passives,
            dev_bboxes=dev_bboxes,
        )
        t2_blockers.extend(shield_blockers)

    all_blockers = tuple([*t1_blockers, *t2_blockers])
    t1_passed = not t1_blockers
    t2_passed = (not t2_blockers) and (not timed_out)
    passed = t1_passed and t2_passed
    rec_expand = max((b.min_expand_nm for b in all_blockers), default=0)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    if passed:
        summary = (
            f"Pre-route gate PASSED in {elapsed_ms:.1f}ms "
            f"(T1=pass, T2=pass, via_redundancy={vr})"
        )
    else:
        top = all_blockers[0] if all_blockers else None
        summary = (
            f"Pre-route gate BLOCKED ({len(all_blockers)} blocker(s), "
            f"recommended_expand_nm={rec_expand}, elapsed={elapsed_ms:.1f}ms): "
            f"{top.reason if top else 'budget_timeout'}"
        )

    n_pins = sum(len(v) for v in (resolved_pins or {}).values())
    return PreRouteGateResult(
        passed=passed,
        t1_passed=t1_passed,
        t2_passed=t2_passed,
        elapsed_ms=elapsed_ms,
        budget_sec=float(budget_sec),
        timed_out=timed_out,
        blockers=all_blockers,
        recommended_expand_nm=rec_expand,
        summary=summary,
        metrics={
            "pdk_id": pdk_id,
            "pex_backend": pex_norm,
            "cap_option": cap_opt_norm,
            "via_redundancy": vr,
            "group_route_contract_present": group_route_contract is not None,
            "n_nets": len(resolved_pins or {}),
            "n_pins": n_pins,
            **corridor_metrics,
        },
    )


__all__ = [
    "PreRouteBlocker",
    "PreRouteGateResult",
    "run_preroute_gate",
]
