"""2.5D shielded geometric coupling-capacitance estimator for matched-pair tuning.

Soft ``matched_cc_balance`` budgets lower to ``Length(..., hardness=SOFT)``.
When two matched nets route in a multi-net layout alongside power rails and
other signal aggressors, pure unmerged wire-length sums diverge systematically
from extracted PEX coupling capacitance (e.g., upper metals couple mainly to
substrate rather than signals, while lower metals couple strongly to crossed
power/signal lines, and staircase via cuts inflate segment-sum length without
adding real area).  This module computes a deterministic, PDK-data-driven 2.5D
shielded metal-stack coupling score directly from committed routes and fixed
conductor literals so ``_materialize_length_constraints`` can optimize real
parasitic balance safely without hardcoding any circuit or net names.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import klayout.db as kdb

from clay.layout.pex.symmetry_balance import DEFAULT_GROUND_NETS

Rect = Tuple[int, int, int, int]

_HS_CLOSE_NM = 175  # 0.35 um effective symmetric spacing envelope (half = 175 nm)
_HS_MED_NM = 375    # 0.75 um effective symmetric spacing envelope (half = 375 nm)
_FRINGE_HALO_NM = 450
_OUTER_HALO_NM = 950
_NM2_TO_UM2 = 1e-6
_DEVICE_PIN_BASE_SCORE = 350.0


def _ground_nets_lower(engine) -> FrozenSet[str]:
    """Lowercased ground-net identifiers (shielded, excluded from PEX coupling)."""
    names = {str(n).strip().casefold() for n in DEFAULT_GROUND_NETS}
    tech = getattr(engine, "_tech", None)
    pdk = getattr(tech, "pdk", None) if tech is not None else None
    sub_fn = getattr(pdk, "substrate_net", None) if pdk is not None else None
    if callable(sub_fn):
        try:
            sub = sub_fn()
            if sub:
                names.add(str(sub).strip().casefold())
        except Exception:
            pass
    return frozenset(names)


def _layer_stack_geometry(engine) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Derive vertical metal heights (um) and relative lateral thickness weights."""
    n_layers = max(1, int(getattr(engine, "_n_layers", 4)))
    spaces = tuple(int(s) for s in getattr(engine, "_space", (140,) * n_layers))
    base_sp = max(1, spaces[0]) if spaces else 140
    z_list = []
    thick_list = []
    z_cur = 1.50
    for gi in range(n_layers):
        sp = spaces[gi] if gi < len(spaces) else spaces[-1]
        if gi > 0:
            step = 0.65 + 0.15 * (gi - 1) + (0.20 if (gi >= 3 and sp > base_sp) else 0.0)
            z_cur += step
        z_list.append(z_cur)
        if sp <= base_sp:
            thick_list.append(0.90)
        elif gi == 2:
            thick_list.append(1.45)
        else:
            thick_list.append(1.25)
    return tuple(z_list), tuple(thick_list)


def _boxes_region(rects: Iterable[Rect]) -> kdb.Region:
    """Build a merged KLayout Region from integer-nm bounding rectangles."""
    reg = kdb.Region()
    for x0, y0, x1, y1 in rects:
        ax, bx = (int(x0), int(x1)) if int(x0) <= int(x1) else (int(x1), int(x0))
        ay, by = (int(y0), int(y1)) if int(y0) <= int(y1) else (int(y1), int(y0))
        if ax < bx and ay < by:
            reg.insert(kdb.Box(ax, ay, bx, by))
    reg.merge()
    return reg


def has_third_party_non_ground_conductors(
    engine,
    committed: Mapping[str, object],
    net_a: str,
    net_b: str,
) -> bool:
    """True when non-ground nets/literals beyond (net_a, net_b) exist to couple against."""
    gnd = _ground_nets_lower(engine)
    pair = {str(net_a), str(net_b)}
    for net in committed:
        s = str(net)
        if s not in pair and s.strip().casefold() not in gnd:
            return True
    for _layer, owner, _rect in getattr(engine, "_fixed_deck_literal_rects", ()):
        if owner is None:
            continue
        s = str(owner)
        if s not in pair and s.strip().casefold() not in gnd:
            return True
    return False


def _score_net_from_regions(
    net_regs: Mapping[int, kdb.Region],
    other_non_vss: Mapping[int, kdb.Region],
    all_shield: Mapping[int, kdb.Region],
    z_um: Sequence[float],
    thick: Sequence[float],
    n_layers: int,
    device_base: float,
) -> float:
    """Evaluate 2.5D lateral + downward/upward shielded coupling for one net."""
    score = float(device_base)
    for gi in range(n_layers):
        rn = net_regs.get(gi)
        if rn is None or rn.is_empty():
            continue
        lw = thick[gi]
        ro_same = other_non_vss.get(gi)
        if ro_same is not None and not ro_same.is_empty():
            for hs_nm, w in ((_HS_CLOSE_NM, 22.0), (_HS_MED_NM, 10.0)):
                ov_um2 = (rn.sized(hs_nm) & ro_same.sized(hs_nm)).area() * _NM2_TO_UM2
                score += lw * w * ov_um2

        un = rn
        for gj in range(gi - 1, -1, -1):
            if un.is_empty():
                break
            sh_gj = all_shield.get(gj)
            nv_gj = other_non_vss.get(gj)
            if nv_gj is not None and not nv_gj.is_empty():
                dz = max(0.15, z_um[gi] - z_um[gj])
                w_v = 1.0 / (dz ** 1.3)
                direct_reg = un & nv_gj
                d0 = direct_reg.area() * _NM2_TO_UM2
                f1_reg = un.sized(_FRINGE_HALO_NM) & nv_gj
                h1 = f1_reg.area() * _NM2_TO_UM2 - d0
                f2_reg = un.sized(_OUTER_HALO_NM) & nv_gj
                h2 = f2_reg.area() * _NM2_TO_UM2 - d0 - h1
                score += lw * w_v * (d0 * 70.0 + max(0.0, h1) * 24.0 + max(0.0, h2) * 9.0)
            if sh_gj is not None and not sh_gj.is_empty():
                un = un - sh_gj

        # Do NOT credit unshielded area over empty field/substrate: native PEX
        # matched_cc_balance excludes VSS/substrate coupling (c_sub).
        un_up = rn
        for gj in range(gi + 1, n_layers):
            if un_up.is_empty():
                break
            sh_gj = all_shield.get(gj)
            nv_gj = other_non_vss.get(gj)
            if nv_gj is not None and not nv_gj.is_empty():
                dz = max(0.15, z_um[gj] - z_um[gi])
                w_v = 1.0 / (dz ** 1.3)
                direct_reg = un_up & nv_gj
                d0 = direct_reg.area() * _NM2_TO_UM2
                f1_reg = un_up.sized(_FRINGE_HALO_NM) & nv_gj
                h1 = f1_reg.area() * _NM2_TO_UM2 - d0
                f2_reg = un_up.sized(_OUTER_HALO_NM) & nv_gj
                h2 = f2_reg.area() * _NM2_TO_UM2 - d0 - h1
                score += thick[gj] * w_v * (d0 * 70.0 + max(0.0, h1) * 24.0 + max(0.0, h2) * 9.0)
            if sh_gj is not None and not sh_gj.is_empty():
                un_up = un_up - sh_gj
    return score


class CouplingContext:
    """Pre-merged per-layer KLayout context for fast candidate coupling scoring."""

    def __init__(
        self,
        engine,
        committed: Mapping[str, object],
        net_a: str,
        net_b: str,
    ) -> None:
        self._engine = engine
        self._net_a = str(net_a)
        self._net_b = str(net_b)
        self._n_layers = max(1, int(getattr(engine, "_n_layers", 4)))
        self._z_um, self._thick = _layer_stack_geometry(engine)
        gnd = _ground_nets_lower(engine)
        tech = getattr(engine, "_tech", None)
        abs_rects = getattr(engine, "_abstract_rects", {}) or {}

        fixed_by_net_gi: Dict[str, Dict[int, list]] = {}
        unowned_gi: Dict[int, list] = {gi: [] for gi in range(self._n_layers)}
        for layer, owner, rect in getattr(engine, "_fixed_deck_literal_rects", ()):
            gi = tech.resolve_layer_to_gi(tuple(layer)) if tech is not None else None
            if gi is None or gi < 0 or gi >= self._n_layers:
                continue
            rn = tuple(int(v) for v in rect)
            if rn in abs_rects.get(gi, ()):
                continue
            if owner is None:
                unowned_gi[gi].append(rn)
            else:
                fixed_by_net_gi.setdefault(str(owner), {}).setdefault(gi, []).append(rn)

        self._fixed_a_rects: Dict[int, Tuple[Rect, ...]] = {
            gi: tuple(fixed_by_net_gi.get(self._net_a, {}).get(gi, ()))
            for gi in range(self._n_layers)
        }
        self._fixed_b_rects: Dict[int, Tuple[Rect, ...]] = {
            gi: tuple(fixed_by_net_gi.get(self._net_b, {}).get(gi, ()))
            for gi in range(self._n_layers)
        }

        has_owned_literals = bool(fixed_by_net_gi.get(self._net_a) or fixed_by_net_gi.get(self._net_b))
        self._device_base = _DEVICE_PIN_BASE_SCORE if has_owned_literals else 0.0

        rest_boxes: Dict[int, list] = {gi: [] for gi in range(self._n_layers)}
        vss_boxes: Dict[int, list] = {gi: list(unowned_gi[gi]) for gi in range(self._n_layers)}
        all_nets = set(committed) | set(fixed_by_net_gi)
        for net in all_nets:
            s = str(net)
            is_gnd = s.strip().casefold() in gnd
            if not is_gnd and s in (self._net_a, self._net_b):
                continue
            target = vss_boxes if is_gnd else rest_boxes
            for gi, rects in fixed_by_net_gi.get(s, {}).items():
                target[gi].extend(rects)
            rn_obj = committed.get(s)
            if rn_obj is not None:
                metal, _cuts = engine._net_metal(rn_obj)
                for gi, rects in metal.items():
                    if 0 <= gi < self._n_layers:
                        target[gi].extend(rects)

        self._rest_non_vss: Dict[int, kdb.Region] = {
            gi: _boxes_region(rest_boxes[gi]) for gi in range(self._n_layers)
        }
        self._vss: Dict[int, kdb.Region] = {
            gi: _boxes_region(vss_boxes[gi]) for gi in range(self._n_layers)
        }
        self._base_regs_a = self._build_net_regions(committed.get(self._net_a), self._fixed_a_rects)
        self._base_regs_b = self._build_net_regions(committed.get(self._net_b), self._fixed_b_rects)
        self._base_other_a = {
            gi: self._rest_non_vss[gi] + self._base_regs_b[gi]
            for gi in range(self._n_layers)
        }
        self._base_shield_a = {
            gi: self._base_other_a[gi] + self._vss[gi]
            for gi in range(self._n_layers)
        }
        self._base_other_b = {
            gi: self._rest_non_vss[gi] + self._base_regs_a[gi]
            for gi in range(self._n_layers)
        }
        self._base_shield_b = {
            gi: self._base_other_b[gi] + self._vss[gi]
            for gi in range(self._n_layers)
        }

    def _build_net_regions(
        self,
        routed_net,
        fixed_rects_by_gi: Mapping[int, Tuple[Rect, ...]],
        only_gis: Optional[Iterable[int]] = None,
        fallback_regs: Optional[Mapping[int, kdb.Region]] = None,
    ) -> Dict[int, kdb.Region]:
        metal: Dict[int, list] = {}
        if routed_net is not None:
            metal, _cuts = self._engine._net_metal(routed_net)
        gis = range(self._n_layers) if only_gis is None else only_gis
        out: Dict[int, kdb.Region] = {} if fallback_regs is None else dict(fallback_regs)
        for gi in gis:
            rects = list(fixed_rects_by_gi.get(gi, ()))
            rects.extend(metal.get(gi, ()))
            out[gi] = _boxes_region(rects)
        return out

    def evaluate(
        self,
        override_net: Optional[str] = None,
        override_route=None,
        touched_gis: Optional[Iterable[int]] = None,
    ) -> Tuple[float, float, float]:
        """Return ``(relative_imbalance, cap_a, cap_b)`` under optional route override."""
        touched_set = set(range(self._n_layers)) if touched_gis is None else set(touched_gis)
        regs_a = self._base_regs_a
        regs_b = self._base_regs_b
        other_for_a = dict(self._base_other_a)
        shield_for_a = dict(self._base_shield_a)
        other_for_b = dict(self._base_other_b)
        shield_for_b = dict(self._base_shield_b)
        if override_net == self._net_a and override_route is not None:
            regs_a = self._build_net_regions(
                override_route, self._fixed_a_rects, touched_set, self._base_regs_a
            )
            for gi in touched_set:
                other_for_b[gi] = self._rest_non_vss[gi] + regs_a[gi]
                shield_for_b[gi] = other_for_b[gi] + self._vss[gi]
        elif override_net == self._net_b and override_route is not None:
            regs_b = self._build_net_regions(
                override_route, self._fixed_b_rects, touched_set, self._base_regs_b
            )
            for gi in touched_set:
                other_for_a[gi] = self._rest_non_vss[gi] + regs_b[gi]
                shield_for_a[gi] = other_for_a[gi] + self._vss[gi]

        cap_a = _score_net_with_rails_and_substrate(
            regs_a, shield_for_a,
            self._z_um, self._thick, self._n_layers, self._device_base,
        )
        cap_b = _score_net_with_rails_and_substrate(
            regs_b, shield_for_b,
            self._z_um, self._thick, self._n_layers, self._device_base,
        )
        denom = max(cap_a, cap_b)
        imbal = 0.0 if denom <= 0.0 else abs(cap_a - cap_b) / denom
        return float(imbal), float(cap_a), float(cap_b)


_OXIDE_EPS_AF_PER_UM = 34.5
_SUBSTRATE_COUPLE_SCALE = 0.30
_PERIM_FRINGE_RATIO = 0.55
_LATERAL_COUPLE_SCALE = 0.30
_FRINGE_SHIELD_FACTOR = 0.35


def _vertical_overlap_cap(
    rg: kdb.Region,
    shield: Dict[int, kdb.Region],
    gi: int,
    layer_indices: range,
    z_um: Tuple[float, ...],
    thick_gi: float,
) -> float:
    """Line-of-sight inter-metal parallel-plate overlap capacitance along one vertical half-space."""
    covered = kdb.Region()
    total = 0.0
    for gj in layer_indices:
        ov = (rg & shield[gj]) - covered
        if not ov.is_empty():
            dz = max(0.25, abs(z_um[gi] - z_um[gj]) - 0.5 * thick_gi)
            total += _OXIDE_EPS_AF_PER_UM * (ov.area() * 1e-6) / dz
        covered += shield[gj]
    return total


def _score_net_with_rails_and_substrate(
    regs: Dict[int, kdb.Region],
    shield: Dict[int, kdb.Region],
    z_um: Tuple[float, ...],
    thick: Tuple[float, ...],
    n_layers: int,
    device_base: float,
) -> float:
    """Estimate total net capacitance including power-rail metal and unshielded substrate/fringe.

    Native PEX checks sum coupling to both signal nets and metal power rails
    (VDD/VSS) plus shielded area/perimeter terms; scoring against non-power
    metal alone under-weights rail couplings and inflates dense lower-layer
    adjacent pin halos relative to higher-layer routes.
    """
    total = float(device_base)
    for gi in range(n_layers):
        rg = regs[gi]
        if rg.is_empty():
            continue
        sh_below = kdb.Region()
        for gj in range(gi):
            sh_below += shield[gj]
        sh_above = kdb.Region()
        for gj in range(gi + 1, n_layers):
            sh_above += shield[gj]
        self_other = kdb.Region()
        for gj in range(n_layers):
            if gj != gi:
                self_other += regs[gj]

        unsh_area_um2 = (rg - sh_below).area() * 1e-6
        total += _SUBSTRATE_COUPLE_SCALE * _OXIDE_EPS_AF_PER_UM * unsh_area_um2 / z_um[gi]

        rg_area = max(1, rg.area())
        other_ov_frac = min(1.0, (rg & (sh_below + sh_above)).area() / rg_area)
        self_ov_frac = min(1.0, (rg & self_other).area() / rg_area)
        eff_perim_um = (rg.perimeter() * 1e-3) * max(
            0.0, 1.0 - _FRINGE_SHIELD_FACTOR * (other_ov_frac + self_ov_frac)
        )
        total += _SUBSTRATE_COUPLE_SCALE * _PERIM_FRINGE_RATIO * _OXIDE_EPS_AF_PER_UM * eff_perim_um

        total += _vertical_overlap_cap(rg, shield, gi, range(gi - 1, -1, -1), z_um, thick[gi])
        total += _vertical_overlap_cap(rg, shield, gi, range(gi + 1, n_layers), z_um, thick[gi])

        lat_um2 = ((rg.sized(350) - rg) & shield[gi]).area() * 1e-6
        total += _LATERAL_COUPLE_SCALE * _OXIDE_EPS_AF_PER_UM * (thick[gi] / 0.25) * (lat_um2 / 0.35)
    return total
