"""PDK layer-stack profile for PEX backends.

Parses Cadence ICT, external_lvs procfile, and external_pdk interconnect-structure data
into a unified ``PdkStackProfile`` that feeds Magic .magicrc, KLayout PEX,
and the geometric estimator with per-PDK physical constants.

ICT format is the richest source (per-layer height/thickness/resistivity/
dielectric-constant); procfile gives deposition-step thickness and ILD k;
external_pdk interconnect tables provide field-solver-calibrated cap coefficients
that serve as the cross-validation oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConductorProfile:
    """One conducting layer in the BEOL/active stack."""

    name: str
    height_um: float          # bottom of layer from substrate (ICT ``height``)
    thickness_um: float       # physical thickness
    resistivity_ohm_um: float  # bulk resistivity (ohm·um)
    min_width_um: float
    min_spacing_um: float
    temp_tc1: float = 0.0     # 1st-order TCR (1/K)
    temp_tc2: float = 0.0     # 2nd-order TCR (1/K^2)
    gate_forming: bool = False
    sheet_rho_ohm_sq: float = 0.0  # computed: resistivity / thickness
    resistivity_width_table: tuple[tuple[float, float], ...] = ()  # (width_um, ohm*um)
    resistivity_thickness_width_table: tuple[tuple[float, tuple[tuple[float, float], ...]], ...] = ()
    temp_tc1_width_table: tuple[tuple[float, float], ...] = ()
    temp_tc2_width_table: tuple[tuple[float, float], ...] = ()

    def resistivity_at_width(self, width_um: float) -> float:
        if not self.resistivity_width_table:
            return self.resistivity_ohm_um
        return _interpolate_resistivity(self.resistivity_width_table, width_um)

    def __post_init__(self):
        if self.sheet_rho_ohm_sq == 0.0 and self.thickness_um > 0:
            object.__setattr__(
                self,
                "sheet_rho_ohm_sq",
                self.resistivity_ohm_um / self.thickness_um,
            )


@dataclass(frozen=True)
class DielectricProfile:
    """One inter-layer dielectric region."""

    name: str
    height_um: float          # bottom of dielectric from substrate
    thickness_um: float       # physical thickness
    dielectric_constant: float  # k (relative permittivity)
    conformal: bool = False


@dataclass(frozen=True)
class PdkStackProfile:
    """Complete per-PDK layer-stack description for PEX backends."""

    pdk_id: str
    node_nm: int
    substrate_height_um: float = 0.0
    conductors: tuple[ConductorProfile, ...] = ()
    conductor_variants: tuple[ConductorProfile, ...] = ()
    dielectrics: tuple[DielectricProfile, ...] = ()
    via_resistance: Dict[Tuple[str, str], float] = field(default_factory=dict)
    mim_f_um2: Optional[float] = None        # MIM capacitance density
    source_files: tuple[str, ...] = ()        # relative paths of parsed data sources
    source_format: str = "none"               # "ict", "procfile", "interconnect", "mixed"
    calibrated: bool = False                  # electrical constants come from foundry data
    warnings: tuple[str, ...] = ()

    # ---- derived ----

    @property
    def conductor_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.conductors)

    def conductor(self, name: str) -> Optional[ConductorProfile]:
        for c in self.conductors:
            if c.name == name:
                return c
        return None

    def dielectric_between(
        self, lower: str, upper: str
    ) -> Optional[DielectricProfile]:
        """Return the (composite) dielectric profile between two conductors."""
        c_lower = self.conductor(lower)
        c_upper = self.conductor(upper)
        if c_lower is None or c_upper is None:
            return None
        lower_top = c_lower.height_um + c_lower.thickness_um
        upper_bot = c_upper.height_um
        gap = upper_bot - lower_top
        if gap <= 0:
            return None
        # Weighted-average dielectric constant of all dielectrics in the gap
        total_kh = 0.0
        for d in self.dielectrics:
            d_bot = max(d.height_um, lower_top)
            d_top = min(d.height_um + d.thickness_um, upper_bot)
            if d_top > d_bot:
                total_kh += d.dielectric_constant * (d_top - d_bot)
        k_eff = total_kh / gap if gap > 0 else 4.0
        return DielectricProfile(
            name=f"{lower}_to_{upper}",
            height_um=lower_top,
            thickness_um=gap,
            dielectric_constant=k_eff,
        )

    @property
    def metal_layers(self) -> tuple[ConductorProfile, ...]:
        return tuple(c for c in self.conductors if c.name.startswith("metal"))

    @property
    def top_metal_name(self) -> str:
        metals = self.metal_layers
        return metals[-1].name if metals else "metal1"


# ---------------------------------------------------------------------------
# ICT parser (Cadence Interconnect Technology format)
# ---------------------------------------------------------------------------


def _strip_comments(line: str) -> str:
    """Remove ICT-style comments (``//`` and ``#``)."""
    for marker in ("//", "#"):
        idx = line.find(marker)
        if idx >= 0:
            return line[:idx]
    return line


def _parse_float(value: str) -> float:
    value = value.strip().rstrip(",")
    # Handle D exponent format: 1.8000E-01
    return float(value.replace("D", "E").replace("d", "e"))


def _parse_ict_block(text: str, block_name: str) -> List[Dict[str, object]]:
    """Parse Cadence ICT ``conductor``/``dielectric`` blocks.

    Returns a list of dicts, one per block.
    """
    import re

    blocks: List[Dict[str, object]] = []
    pattern = re.compile(
        rf"^\s*{re.escape(block_name)}\s+(\S+)\s*\{{",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(text):
        name = match.group(1).strip().strip('"')
        open_brace = match.end() - 1
        depth = 0
        close_brace = None
        for index in range(open_brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    close_brace = index
                    break
        if close_brace is None:
            continue
        body = text[open_brace + 1 : close_brace]
        entry: Dict[str, object] = {"name": name}
        nested_depth = 0
        last_key = None
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            opens = line.count("{")
            closes = line.count("}")
            if nested_depth > 0 or opens:
                nested_depth += opens - closes
                continue
            # ICT matrices continue on numeric-leading rows. Retain all rows
            # under their owning field instead of treating their first value
            # as a new dictionary key (which also loses repeated first values).
            if re.match(r"^[+\-]?(?:\d|\.\d)", line) and last_key is not None:
                entry[last_key] = str(entry[last_key]) + " " + line.rstrip(",")
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key = parts[0].strip()
                last_key = key
                val = parts[1].strip().rstrip(",")
                try:
                    if key in (
                        "min_spacing",
                        "min_width",
                        "height",
                        "thickness",
                        "resistivity",
                        "temp_tc1",
                        "temp_tc2",
                        "dielectric_constant",
                    ):
                        entry[key] = _parse_float(val)
                    elif key in ("gate_forming_layer", "conformal"):
                        entry[key] = val.lower() == "true"
                    elif key == "capacitor_only_layer_to":
                        entry[key] = val
                    elif key == "layer_type":
                        entry[key] = val
                    else:
                        entry[key] = val
                except ValueError:
                    entry[key] = val
        blocks.append(entry)
    return blocks


def _parse_ict_subconductors(
    text: str, parents: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Return nested ICT sub-conductors with missing geometry inherited."""
    variants = _parse_ict_block(text, "sub_conductor")
    for variant in variants:
        name = str(variant.get("name", "")).lower()
        parent = next(
            (
                candidate
                for candidate in parents
                if name.endswith(str(candidate.get("name", "")).lower())
            ),
            None,
        )
        if parent is None:
            continue
        for key, value in parent.items():
            variant.setdefault(key, value)
    return variants


def _parse_ict_via_resistances(text: str) -> Dict[Tuple[str, str], float]:
    """Read top-level ICT ``via`` blocks and their foundry contact resistance."""
    import re

    lines = text.splitlines()
    via_resistance: Dict[Tuple[str, str], float] = {}
    index = 0
    while index < len(lines):
        if re.match(r"^\s*via\s+\S+\s*\{", lines[index]) is None:
            index += 1
            continue
        block = [lines[index]]
        depth = lines[index].count("{") - lines[index].count("}")
        index += 1
        while index < len(lines) and depth > 0:
            block.append(lines[index])
            depth += lines[index].count("{") - lines[index].count("}")
            index += 1
        body = "\n".join(block)
        lower = re.search(r"^\s*bottom_layer\s+(\S+)", body, re.MULTILINE)
        upper = re.search(r"^\s*top_layer\s+(\S+)", body, re.MULTILINE)
        resistance = re.search(
            r"^\s*contact_resistance\s+(\S+)", body, re.MULTILINE
        )
        if lower is not None and upper is not None and resistance is not None:
            via_resistance[(lower.group(1), upper.group(1))] = _parse_float(
                resistance.group(1)
            )
    return via_resistance


def _interpolate_resistivity(table, width_um):
    if width_um <= table[0][0]:
        return table[0][1]
    for (w0, rho0), (w1, rho1) in zip(table, table[1:]):
        if width_um <= w1:
            return rho0 + (rho1 - rho0) * (width_um - w0) / (w1 - w0)
    return table[-1][1]


def _ict_resistivity(raw):
    if "rho_values" in raw:
        widths = tuple(_parse_float(v) for v in str(raw["rho_silicon_widths"]).split())
        thicknesses = tuple(_parse_float(v) for v in str(raw["rho_silicon_thicknesses"]).split())
        values = tuple(_parse_float(v) for v in str(raw["rho_values"]).split())
        if (not widths or not thicknesses or len(values) != len(widths) * len(thicknesses)
                or any(v <= 0 for v in (*widths, *thicknesses, *values))
                or any(a >= b for axis in (widths, thicknesses) for a, b in zip(axis, axis[1:]))):
            raise ValueError("ICT native resistivity matrix has invalid axes or dimensions")
        surface = tuple((thickness, tuple(zip(widths, values[i * len(widths):(i + 1) * len(widths)])))
                        for i, thickness in enumerate(thicknesses))
        thickness = float(raw.get("thickness", .1))
        table = tuple((width, _interpolate_resistivity(tuple((t, row[j][1]) for t, row in surface), thickness))
                      for j, width in enumerate(widths))
        return _interpolate_resistivity(table, float(raw.get("min_width", .2))), table, surface
    value = raw.get("resistivity", 0.1)
    if isinstance(value, (float, int)):
        return float(value), (), ()
    values = tuple(_parse_float(token) for token in str(value).replace(",", " ").split())
    if len(values) == 1:
        return values[0], (), ()
    if not values or len(values) % 2:
        raise ValueError("ICT resistivity requires a scalar or (resistivity,width) pairs")
    table = tuple((values[i + 1], values[i]) for i in range(0, len(values), 2))
    if (any(w <= 0 or rho <= 0 for w, rho in table)
            or any(a[0] >= b[0] for a, b in zip(table, table[1:]))):
        raise ValueError("ICT resistivity width table must be positive and strictly increasing")
    return _interpolate_resistivity(table, float(raw.get("min_width", 0.2))), table, ()


def _ict_width_parameter(raw, key, default, resistivity_table):
    value = raw.get(key, default)
    if isinstance(value, (float, int)):
        return float(value), ()
    values = tuple(_parse_float(token) for token in str(value).replace(",", " ").split())
    if len(values) == 1:
        return values[0], ()
    widths = (tuple(_parse_float(v) for v in str(raw["temp_widths"]).split())
              if "temp_widths" in raw else tuple(item[0] for item in resistivity_table))
    if (not widths or len(values) != len(widths) or any(w <= 0 for w in widths)
            or any(a >= b for a, b in zip(widths, widths[1:]))):
        raise ValueError(f"ICT {key} table must align with its native temperature width grid")
    table = tuple(zip(widths, values))
    return _interpolate_resistivity(table, float(raw.get("min_width", .2))), table


def parse_ict(path: str | Path, pdk_id: str = "", node_nm: int = 180) -> PdkStackProfile:
    """Parse a Cadence ICT file into ``PdkStackProfile``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    conductors_raw = _parse_ict_block(text, "conductor")
    variants_raw = _parse_ict_subconductors(text, conductors_raw)
    dielectrics_raw = _parse_ict_block(text, "dielectric")

    # Parse process header
    import re
    process_match = re.search(r"process\s+(\S+)\s*\{", text)
    process_name = process_match.group(1) if process_match else ""
    bg_k = 1.0
    temp_ref = 25.0
    if process_match:
        proc_body = text[process_match.start() : text.find("}", process_match.start())]
        for m in re.finditer(r"background_dielectric_constant\s+([\d.]+)", proc_body):
            bg_k = float(m.group(1))
        for m in re.finditer(r"temp_reference\s+([\d.]+)", proc_body):
            temp_ref = float(m.group(1))

    conductors: List[ConductorProfile] = []
    for raw in conductors_raw:
        name = str(raw.get("name", ""))
        resistivity, resistivity_table, resistivity_surface = _ict_resistivity(raw)
        tc1, tc1_table = _ict_width_parameter(raw, "temp_tc1", 0., resistivity_table)
        tc2, tc2_table = _ict_width_parameter(raw, "temp_tc2", 0., resistivity_table)
        c = ConductorProfile(
            name=name,
            height_um=float(raw.get("height", 0.0)),
            thickness_um=float(raw.get("thickness", 0.1)),
            resistivity_ohm_um=resistivity,
            resistivity_width_table=resistivity_table,
            resistivity_thickness_width_table=resistivity_surface,
            min_width_um=float(raw.get("min_width", 0.2)),
            min_spacing_um=float(raw.get("min_spacing", 0.2)),
            temp_tc1=tc1, temp_tc1_width_table=tc1_table,
            temp_tc2=tc2, temp_tc2_width_table=tc2_table,
            gate_forming=bool(raw.get("gate_forming_layer", False)),
        )
        conductors.append(c)

    conductor_variants: List[ConductorProfile] = []
    for raw in variants_raw:
        resistivity, resistivity_table, resistivity_surface = _ict_resistivity(raw)
        tc1, tc1_table = _ict_width_parameter(raw, "temp_tc1", 0., resistivity_table)
        tc2, tc2_table = _ict_width_parameter(raw, "temp_tc2", 0., resistivity_table)
        conductor_variants.append(
            ConductorProfile(
                name=str(raw.get("name", "")),
                height_um=float(raw.get("height", 0.0)),
                thickness_um=float(raw.get("thickness", 0.1)),
                resistivity_ohm_um=resistivity,
                resistivity_width_table=resistivity_table,
                resistivity_thickness_width_table=resistivity_surface,
                min_width_um=float(raw.get("min_width", 0.2)),
                min_spacing_um=float(raw.get("min_spacing", 0.2)),
                temp_tc1=tc1, temp_tc1_width_table=tc1_table,
                temp_tc2=tc2, temp_tc2_width_table=tc2_table,
                gate_forming=bool(raw.get("gate_forming_layer", False)),
            )
        )

    dielectrics: List[DielectricProfile] = []
    for raw in dielectrics_raw:
        d = DielectricProfile(
            name=str(raw.get("name", "")),
            height_um=float(raw.get("height", 0.0)),
            thickness_um=float(raw.get("thickness", 0.1)),
            dielectric_constant=float(raw.get("dielectric_constant", 4.0)),
            conformal=bool(raw.get("conformal", False)),
        )
        dielectrics.append(d)

    # Sort by height
    conductors.sort(key=lambda c: c.height_um)
    dielectrics.sort(key=lambda d: d.height_um)

    warnings: List[str] = []
    if any(c.resistivity_width_table for c in (*conductors, *conductor_variants)):
        warnings.append("width-dependent resistivity tables retained; scalar sheet resistance is evaluated at min_width")
    if bg_k != 1.0:
        warnings.append(f"background_dielectric_constant={bg_k} (not used in planar model)")

    via_resistance = _parse_ict_via_resistances(text)
    if not via_resistance:
        warnings.append("ICT contains no foundry contact_resistance values")

    # Detect MIM
    mim_f_um2: Optional[float] = None
    for raw in conductors_raw:
        if raw.get("capacitor_only_layer_to"):
            mim_name = str(raw.get("name", ""))
            target = str(raw.get("capacitor_only_layer_to", ""))
            mim_cond = None
            target_cond = None
            for c in conductors:
                if c.name == mim_name:
                    mim_cond = c
                if c.name == target:
                    target_cond = c
            if mim_cond and target_cond:
                gap = mim_cond.height_um - (target_cond.height_um + target_cond.thickness_um)
                k_eff = 4.0
                for d in dielectrics:
                    d_bot = max(d.height_um, target_cond.height_um + target_cond.thickness_um)
                    d_top = min(d.height_um + d.thickness_um, mim_cond.height_um)
                    if d_top > d_bot:
                        k_eff = d.dielectric_constant
                eps0 = 8.8541878128e-3  # fF/um
                mim_f_um2 = round(eps0 * k_eff / gap, 4)

    return PdkStackProfile(
        pdk_id=pdk_id or process_name or path.stem,
        node_nm=node_nm,
        conductors=tuple(conductors),
        conductor_variants=tuple(conductor_variants),
        dielectrics=tuple(dielectrics),
        via_resistance=via_resistance,
        mim_f_um2=mim_f_um2,
        source_files=(str(path),),
        source_format="ict",
        calibrated=True,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Procfile parser (external_lvs RCX format) -- extends existing pdk_constants parser
# ---------------------------------------------------------------------------
# The existing ``pdk_constants._read_stack`` already handles procfile
# deposition steps.  This module re-exposes the results through the
# ``PdkStackProfile`` data model.


def parse_procfile(path: str | Path, pdk_id: str = "", node_nm: int = 180) -> PdkStackProfile:
    """Parse an external_lvs RCX procfile into ``PdkStackProfile``."""
    import re

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    process_match = re.search(r"process\s+(\S+)", text)
    process_name = process_match.group(1) if process_match else pdk_id

    # Parse deposition steps (new parser)

    conductors: List[ConductorProfile] = []
    dielectrics: List[DielectricProfile] = []
    cum_height = 0.0

    lines = text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("*") or line.startswith(";"):
            continue
        if not line.lower().startswith("step"):
            continue
        # Split by whitespace
        parts = line.split()
        if len(parts) < 5:
            continue
        # parts[0]="step", parts[1]=step_num
        step_type = parts[2].lower()
        # Determine name and thickness positions
        if step_type == "metal":
            # step N metal name thickness param [extra]
            if len(parts) < 5:
                continue
            name = parts[3]
            thickness = float(parts[4].replace(",", "."))
            conductors.append(
                ConductorProfile(
                    name=name,
                    height_um=cum_height,
                    thickness_um=thickness,
                    resistivity_ohm_um=0.080 * thickness,
                    min_width_um=0.23,
                    min_spacing_um=0.23,
                )
            )
            cum_height += thickness
        elif step_type == "dielectric":
            # step N dielectric [p/n] name thickness param
            if len(parts) < 6:
                continue
            name = parts[4] if len(parts) >= 6 else parts[3]
            thickness = float(parts[5].replace(",", ".")) if len(parts) >= 6 else float(parts[4].replace(",", "."))
            param = float(parts[6].replace(",", ".")) if len(parts) >= 7 else 4.0
            dielectrics.append(
                DielectricProfile(
                    name=name,
                    height_um=cum_height,
                    thickness_um=thickness,
                    dielectric_constant=param if param > 1.0 else 4.0,
                )
            )
            cum_height += thickness
        elif step_type == "substrate":
            # step N substrate name thickness param
            name = parts[3]
            thickness = float(parts[4].replace(",", "."))
            param = float(parts[5].replace(",", ".")) if len(parts) >= 6 else 3.9
            dielectrics.append(
                DielectricProfile(
                    name=name,
                    height_um=0.0,
                    thickness_um=thickness,
                    dielectric_constant=param if param > 1.0 else 3.9,
                )
            )

    # Parse metal_variations for actual min width/spacing
    mv_pattern = re.compile(
        r"layer\s+(\S+)\s+([ws])\s+([\d.]+)\s+([\d.]+)",
        re.IGNORECASE,
    )
    layer_mins: Dict[str, Dict[str, float]] = {}
    for match in mv_pattern.finditer(text):
        lname = match.group(1)
        ws = match.group(2).lower()
        min_val = float(match.group(3))
        if lname not in layer_mins:
            layer_mins[lname] = {}
        layer_mins[lname][f"min_{ws}"] = min(
            min_val, layer_mins[lname].get(f"min_{ws}", min_val)
        )

    # Apply min dimensions to conductors
    conductors_out: List[ConductorProfile] = []
    for c in conductors:
        mins = layer_mins.get(c.name, {})
        conductors_out.append(
            ConductorProfile(
                name=c.name,
                height_um=c.height_um,
                thickness_um=c.thickness_um,
                resistivity_ohm_um=c.resistivity_ohm_um,
                min_width_um=mins.get("min_w", c.min_width_um),
                min_spacing_um=mins.get("min_s", c.min_spacing_um),
            )
        )

    return PdkStackProfile(
        pdk_id=process_name,
        node_nm=node_nm,
        conductors=tuple(conductors_out),
        dielectrics=tuple(dielectrics),
        via_resistance={},
        source_files=(str(path),),
        source_format="procfile",
        calibrated=False,
        warnings=(
            "sheet resistance uses the explicit uncalibrated 0.080 ohm/sq fallback",
            "via resistance unavailable in procfile; no inferred values emitted",
        ),
    )


# ---------------------------------------------------------------------------
# external_pdk interconnect-structure parser
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Magic .magicrc generator
# ---------------------------------------------------------------------------


def generate_magicrc(profile: PdkStackProfile) -> str:
    """Generate a Magic .magicrc extraction file from a ``PdkStackProfile``.

    The output is a minimal rcfile that sets up the extraction parameters
    (sheet resistances, contact resistances) for Magic's ``ext2spice``.
    It does NOT include DRC rules, device models, or full layer definitions
    — those remain with the foundry PDK; this file is for PEX only.
    """
    lines: List[str] = []
    lines.append(f"# Magic extraction rcfile for {profile.pdk_id}")
    lines.append(f"# Generated from {profile.source_format}: {', '.join(profile.source_files)}")
    lines.append(f"# Node: {profile.node_nm} nm")
    lines.append("")

    # ---- Extract section with sheet resistances ----
    lines.append("extract")
    for c in profile.conductors:
        if c.sheet_rho_ohm_sq > 0:
            # Map to Magic-style type names
            mtype = _magic_type_name(c.name)
            lines.append(f"  resist {mtype} {c.sheet_rho_ohm_sq:.4f}")
    lines.append("")

    # Contact resistances
    for (lower, upper), r in sorted(profile.via_resistance.items()):
        ltype = _magic_type_name(lower)
        utype = _magic_type_name(upper)
        lines.append(f"  contact {ltype},{utype} {r:.2f}")
    lines.append("")

    # ---- Dielectric constants (informational; Magic uses its own) ----
    lines.append("# Dielectric stack (informational — Magic uses its own plane geometry)")
    for d in profile.dielectrics:
        lines.append(f"#  {d.name:30s} h={d.height_um:6.3f} t={d.thickness_um:6.3f} k={d.dielectric_constant:.3f}")
    lines.append("")

    # ---- MIM ----
    if profile.mim_f_um2 is not None:
        lines.append(f"# MIM capacitance density: {profile.mim_f_um2:.4f} fF/um2")
        lines.append("")

    # ---- CIF/GDS layer hints ----
    lines.append("# GDS layer mapping must be provided by the PDK .layermap file.")
    lines.append("# Load this rcfile AFTER the technology's main .tech file:")
    lines.append(f"#   magic -rcfile <pdk>.tech -rcfile {profile.pdk_id}_pex.magicrc")
    lines.append("")

    return "\n".join(lines)


def _magic_type_name(pdk_layer: str) -> str:
    """Heuristic PDK layer name → Magic type name."""
    mapping = {
        "active": "ndiff",
        "nactive": "ndiff",
        "pactive": "pdiff",
        "poly": "poly",
        "npoly": "npoly",
        "ppoly": "ppoly",
        "metal1": "metal1",
        "metal2": "metal2",
        "metal3": "metal3",
        "metal4": "metal4",
        "metal5": "metal5",
        "metal6": "metal6",
        "mim": "mimcap",
        "contact": "polycont",
        "via1": "via1",
        "via2": "via2",
        "via3": "via3",
        "via4": "via4",
        "via5": "via5",
    }
    return mapping.get(pdk_layer.lower(), pdk_layer.lower())


# ---------------------------------------------------------------------------
# Convenience: load profile for any registered PDK
# ---------------------------------------------------------------------------

_PDK_PROFILES: Dict[str, PdkStackProfile] = {}


def _resolve_pdk_path(pdk_id: str) -> Optional[Path]:
    """Resolve the root directory of a PDK installation.

    Checks (in order): the PDK's own resolver, known subdirectories under
    ``pdks/``, and environment variables.
    """
    import os

    # Try per-PDK resolver
    resolver_map = {
        "gf180mcu": ("clay.pdk.gf180mcu", "resolve_pdk_root"),
    }
    if pdk_id in resolver_map:
        mod_name, func_name = resolver_map[pdk_id]
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            resolver = getattr(mod, func_name, None)
            if resolver:
                result = resolver()
                if result is not None:
                    return result
        except Exception:
            pass

    # Search pdks/ directory (handle worktree → main repo fallback)
    candidate = Path(__file__).resolve()
    repo = None
    for p in candidate.parents:
        if (p / ".git").is_file():  # worktree .git file
            git_file = (p / ".git").read_text(encoding="utf-8", errors="replace").strip()
            if git_file.startswith("gitdir:"):
                gitdir = Path(git_file.split(":", 1)[1].strip())
                repo = gitdir.parent.parent.parent  # .git/worktrees/<name> → main repo
                break
        elif (p / ".git").is_dir():  # regular repo
            repo = p
            break
    if repo is None:
        repo = candidate.parents[1]
    pdks_root = repo / "pdks"
    if pdks_root.is_dir():
        for entry in sorted(pdks_root.iterdir()):
            if entry.is_dir() and pdk_id.lower() in entry.name.lower():
                # For nested layouts (open_pdk/external_pdk/)
                oa = entry / f"{entry.name}_OA"
                if oa.is_dir():
                    return oa
                return entry

    return None


def get_profile(pdk_id: str) -> Optional[PdkStackProfile]:
    """Return a cached or auto-discovered ``PdkStackProfile`` for *pdk_id*."""
    if pdk_id in _PDK_PROFILES:
        return _PDK_PROFILES[pdk_id]

    # Resolve PDK root via the per-PDK resolver function
    root = _resolve_pdk_path(pdk_id)
    if root is None:
        return None
    if root is None:
        return None

    profile = _discover_profile(pdk_id, root)
    if profile is not None:
        _PDK_PROFILES[pdk_id] = profile
    return profile


def _gf180_raphael_table_profile(root: Path) -> Optional[PdkStackProfile]:
    """Read the installed 6LM/9K/MIM-A cross-section, not generic GF constants.

    The foundry cross-section explicitly embeds each metal in its named IMD:
    the 1.15um IMD pitch is NOT a 1.15um empty inter-metal gap. MIM-A expands
    IMD2 to 1.45um. The supported route subset is M1..M5 of the physical 6LM
    option, not a 5LM process with a thick top M5.
    """
    import csv

    docs = root/'docs/analog/layout/inter_specs'
    conductor_file = docs/'tables_clear/3_General_conductor.csv'
    dielectric_file = docs/'tables_clear/4_General_Dielectric.csv'
    cross_section = docs/'images/2_cross_section_16.png'
    if not conductor_file.is_file() or not dielectric_file.is_file():
        return None
    if not cross_section.is_file():
        raise ValueError('GF180 table profile requires its layer-height cross-section authority')

    def rows(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return {row[0].strip(): row for row in list(csv.reader(stream))[1:] if row}

    metals, dielectrics = rows(conductor_file), rows(dielectric_file)
    ild = float(dielectrics['ILD'][1])
    liner = float(dielectrics['Nitr liner'][1])
    dielectric_rows = [
        DielectricProfile('Nitr liner (planarized approximation)', 0., liner,
                          float(dielectrics['Nitr liner'][4]), conformal=True),
        DielectricProfile('ILD', liner, ild-liner, float(dielectrics['ILD'][4])),
    ]
    conductors = []
    z = ild
    for level in range(1, 6):
        row = metals[f'M{level}']
        conductors.append(ConductorProfile(f'metal{level}', z, float(row[1]), 0.,
                                           float(row[4]), float(row[7])))
        dielectric_name = 'IMD2 (with MIM)' if level == 2 else f'IMD{level}'
        dielectric = dielectrics[dielectric_name]
        pitch = float(dielectric[1])
        if pitch <= float(row[1]):
            raise ValueError('GF180 IMD does not leave a positive metal-to-metal gap')
        dielectric_rows.append(DielectricProfile(dielectric_name, z, pitch,
                                                 float(dielectric[4])))
        z += pitch
    top = metals['TM (9KA)']
    top_variant = ConductorProfile('metal6', z, float(top[1]), 0., float(top[4]), float(top[7]))
    return PdkStackProfile(
        pdk_id='gf180mcu', node_nm=180, conductors=tuple(conductors),
        conductor_variants=(top_variant,), dielectrics=tuple(dielectric_rows),
        source_files=tuple(str(path.resolve()) for path in
                           (conductor_file, dielectric_file, cross_section)),
        source_format='raphael-input-tables-6LM-9K-MIM-A', calibrated=False,
        warnings=(
            'Native M1..M5 are the routable subset of 6LM/9K; physical M6 is retained as a non-routable conductor variant',
            'Substrate datum is the Si/STI top; conformal 0.05um nitride is flattened within the 0.86um ILD; field-versus-active substrate and shielding are not modeled',
            'These are native Raphael input dimensions, not a new Raphael solve; resistivity is unspecified and no wire resistance is extracted',
        ),
    )


def _discover_profile(pdk_id: str, root: Path) -> Optional[PdkStackProfile]:
    """Auto-discover and parse the best available stack data for a PDK."""
    # The installed GF process option has explicit native Raphael input tables
    # and a cross-section defining whether an IMD thickness includes metal.
    if pdk_id == 'gf180mcu':
        native = _gf180_raphael_table_profile(root)
        if native is not None:
            return native
    # Priority: ICT > procfile > interconnect tables
    # 1) Look for ICT files
    ict_candidates = sorted(root.rglob("*.ict"))
    if ict_candidates:
        # Prefer TYPICAL corner
        for cand in ict_candidates:
            if "typ" in cand.name.lower():
                return parse_ict(cand, pdk_id=pdk_id)
        return parse_ict(ict_candidates[0], pdk_id=pdk_id)

    # 2) Look for procfile (external_lvs RCX)
    for cand in root.rglob("procfile"):
        return parse_procfile(cand, pdk_id=pdk_id)


    return None
