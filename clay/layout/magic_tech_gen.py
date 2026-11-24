"""Auto-generate a minimal Magic technology file (.tech) from a commercial PDK's
OWN parsed data, so Magic ``ext2spice`` PEX can run on a PDK that ships no Magic
techfile (external_pdk / external_pdk) -- the "fourth path" for parasitic extraction.

WHAT THIS IS / IS NOT
---------------------
Magic only ships a hand-written ``.tech`` for the open PDKs (gf180mcu, sky130).
For a COMMERCIAL PDK there is no Magic tech, so ``MagicPexBackend`` cannot run.
This module synthesises the MINIMAL Magic tech needed to extract a metal1 /
metal2 / poly / contact / via1 layout, entirely from the PDK's OWN data:

  * ``cifinput`` GDS ``calma <layer> <datatype>`` lines come from
    :func:`clay.frontend.pdk_layermap.parse_layermap` -- the REAL drawn-layer GDS
    numbers traced to the PDK's own external_drc deck / KLayout .lyp (nothing guessed).
  * the ``areacap`` / ``perimcap`` / ``sidewall`` / ``defaultoverlap`` /
    ``defaultsideoverlap`` capacitance coefficients are COMPUTED from
    :func:`clay.layout.pex.pdk_constants.resolve_pdk_pex_constants` -- the real
    per-PDK metal1 thickness, inter-metal dielectric k, metal1->metal2 vertical
    gap and min spacing read from the PDK's own interconnect-tech file + DRC deck.

NET MERGING (first-class contacts)
----------------------------------
Contacts and vias are FIRST-CLASS Magic ``contact`` types, not plain drawn
layers, so Magic's scanline extractor MERGES a net through them.  Poly and diff
share one ``active`` plane (the scmos idiom); the single drawn contact layer is
split in ``cifinput`` into a poly-contact (``contact AND poly``) and a
diff-contact (``contact AND diff``), each bridging up to metal1, and the via1
layer becomes a metal1<->metal2 contact.  The metal net LABELS (GDS text on
``(metal_gds, 10)``) are read onto the metal types so a merged net keeps its
design name.  Validated on the real comparator GDS: the 10 design nets extract
as exactly 10 single merged named nodes (di_n / di_p / outm / outp included),
zero per-fragment split -- before this, the same layout split into ~49 per-metal
fragments.  The ``connect`` section keeps poly and diff electrically distinct
(they merge only WHERE a contact bridges them, never blanket-plane).

HONESTY (non-negotiable)
------------------------
A Magic extraction driven by THIS generated tech is::

    "Magic 2.5D from a Clay-generated tech (real per-PDK constants)"

It is NOT the foundry's signed StarRC / QRC ITF deck, and NOT a field solve.
The cap coefficients are a parallel-plate + side-fringe geometric model evaluated
at the PDK's real BEOL dimensions -- the SAME model the geometric backend uses,
just expressed in Magic's tech-file vocabulary so Magic's own scanline extractor
does the area/perimeter/overlap integration over the real polygons. Every emitted
coefficient carries provenance back to ``resolve_pdk_pex_constants`` (see
:func:`tech_provenance`). A generated tech is only claimed USABLE if Magic
actually loads it and ext2spice produces finite, non-zero parasitics on a real
GDS (:func:`run_magic_pex`); if Magic rejects it, the exact error is returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clay.frontend.pdk_layermap import parse_layermap_detail
from clay.layout.pex.pdk_constants import PdkPexConstants, resolve_pdk_pex_constants

# The canonical conductor tokens this minimal tech can place on a Magic plane,
# and the (magic_type, magic_plane) each maps to. We only emit a conductor when
# the PDK layermap actually resolved its GDS number.
#
# CONTACTS ARE FIRST-CLASS (net-merge sound).  ``contact`` (poly/diff -> metal1)
# and ``via1`` (metal1 -> metal2) are NOT plain drawn layers; they are Magic
# ``contact`` types that physically occupy BOTH connected planes (a residue on
# each), so Magic's scanline extractor MERGES a net through them.  Because a
# Magic contact type bridges exactly two planes, the single drawn contact layer
# (which lands on poly OR diff) is split, in ``cifinput``, into two contact types
# via boolean AND with the poly / diff layer:
#
#     pcont = contact_layer AND poly   (poly  <-> metal1)
#     dcont = contact_layer AND diff   (ndiff <-> metal1)
#     m1via = via1_layer               (metal1 <-> metal2)
#
# Poly and diff share ONE ``active`` plane (the scmos idiom) so a contact's lower
# residue can sit on either; the ``connect`` section keeps poly and diff
# electrically distinct (they merge only WHERE a contact bridges them, never
# blanket-plane).  This was validated on the real comparator GDS: the 10 design
# nets extract as 10 single merged named nodes (see tests).
#
#   token   -> (magic_type, magic_plane)
_TOKEN_PLANE: Dict[str, Tuple[str, str]] = {
    "nwell": ("nwell", "well"),
    "diff": ("ndiff", "active"),
    "poly": ("poly", "active"),   # poly + diff share the active plane
    "metal1": ("metal1", "metal1"),
    "metal2": ("metal2", "metal2"),
}

# First-class Magic contact types: name -> (lower_type, upper_type, planes-it-
# bridges via the lower conductor it AND's against in cifinput). ``draw_token`` is
# the conductor token whose GDS shape the contact is AND'ed against; ``src_token``
# is the layermap token carrying the GDS contact/via layer number.
@dataclass(frozen=True)
class _ContactSpec:
    name: str          # magic contact type name (e.g. "pcont")
    lower: str         # magic type on the active plane it connects (poly/ndiff)
    upper: str         # magic type on the upper plane (metal1/metal2)
    src_token: str     # layermap token of the drawn contact/via layer (contact/via1)
    and_token: Optional[str]  # conductor token to AND against (poly/diff) or None


_CONTACTS: List[_ContactSpec] = [
    _ContactSpec("pcont", "poly", "metal1", "contact", "poly"),
    _ContactSpec("dcont", "ndiff", "metal1", "contact", "diff"),
    _ContactSpec("m1via", "metal1", "metal2", "via1", None),
]

# Layermap tokens that are CONTACT/VIA SOURCE layers (drawn cut layers), not
# conductors with a plane of their own. Their GDS numbers feed the first-class
# contact-type painting in cifinput; they never appear in _TOKEN_PLANE.
_CONTACT_SOURCE = {c.src_token for c in _CONTACTS}

# Planes this minimal tech declares (ordered bottom -> top); planeorder follows.
# poly + diff live together on ``active`` so a single drawn contact layer can
# bridge either one up to metal1.
_PLANES: List[Tuple[str, str]] = [
    ("well", "w"),
    ("active", "a"),
    ("metal1", "m1"),
    ("metal2", "m2"),
]

# GDS texttype carrying Clay net labels on a metal layer (the layout writer's
# convention: text on (metal_gds, 10); see clay._ir_bridge_layout_label_rs).
# Reading these onto the metal type is what gives merged nets their design names.
_LABEL_TEXTTYPE: int = 10

# Side-fringe multiplier on the bare parallel-plate side-wall term, mirroring
# clay.layout.pex.coupling.DEFAULT_K_FRINGE (the two over/under fringe paths).
_K_FRINGE: float = 2.0


@dataclass(frozen=True)
class GeneratedTech:
    """A generated Magic tech plus the provenance of every cap coefficient."""

    pdk_id: str
    text: str
    layers: Dict[str, Tuple[int, int]]       # token -> (gds_layer, gds_datatype)
    coeffs: Dict[str, float]                  # name -> aF/um^2 or aF/um
    constants: PdkPexConstants
    layermap_source: Optional[str]


# ---------------------------------------------------------------------------
# Capacitance coefficient model (real per-PDK constants -> Magic aF/um*, aF/um^2)
# ---------------------------------------------------------------------------


def _cap_coeffs(c: PdkPexConstants) -> Dict[str, float]:
    """Derive Magic cap coefficients from the per-PDK geometric constants.

    Units match Magic's ``units microns`` convention: ``areacap`` /
    ``defaultoverlap`` in aF/um^2 (a per-unit-AREA plate cap), ``sidewall`` /
    ``perimcap`` / ``defaultsideoverlap`` in aF/um (a per-unit-LENGTH fringe cap).

    Model (the same parallel-plate + side-fringe physics as
    ``clay.layout.pex.coupling``)::

        area  = eps_ild / vgap                         [aF/um^2]   (vertical plate)
        fringe= eps_ild * (thickness / min_space) * k  [aF/um]     (side fringe)

    ``vgap`` is the real metal1->metal2 vertical separation; ``thickness`` the
    real metal1 conductor thickness; ``min_space`` the real metal1 min spacing;
    all sourced (with provenance) from ``resolve_pdk_pex_constants``.
    """
    eps = c.eps_ild_aF_per_um          # aF/um  (= eps0 * ild_k)
    vgap = c.metal_vgap_um             # um
    t = c.metal_thickness_um           # um
    s = c.min_space_um                 # um

    area = eps / vgap                  # aF/um^2  (eps[aF/um] / um)
    fringe = eps * (t / s) * _K_FRINGE  # aF/um

    return {
        "areacap": area,
        "perimcap": fringe,
        "sidewall": fringe,
        "overlap": area,        # metal-over-metal vertical plate
        "sideoverlap": fringe,  # metal-over-metal side fringe
    }


# ---------------------------------------------------------------------------
# Tech-file section emitters
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _emit_cifinput(layers: Dict[str, Tuple[int, int]]) -> str:
    """``cifinput`` GDS-read style: real (gds, datatype) -> magic type per layer,
    INCLUDING first-class contact types and metal net-label reading.

    Magic's cifinput grammar:

      * ``layer <magictype> <CIFNAME>``  -- magic TYPE first, CIF layer name(s)
        second (the reverse of cifoutput).  A following indented ``and <CIF>`` /
        ``labels <CIF>`` refines that paint: ``and`` intersects (used to split the
        single drawn contact layer into poly-contact vs diff-contact), ``labels``
        attaches GDS text on a CIF layer onto the magic type (so a merged net keeps
        its design name).
      * ``calma <CIFNAME> <gds_layer> <gds_datatype>``  -- a SEPARATE binding block
        (THREE args) mapping each CIF name to its GDS number / texttype.

    A contact type painted here occupies BOTH its planes (Magic forms the
    vertical net connection); ``and POLY`` / ``and DIFF`` keep poly-contacts and
    diff-contacts as distinct types so poly and diff merge only where a real
    contact bridges them.
    """
    lines = [
        "cifinput",
        "style import",
        " scalefactor 1 nanometers",
        " gridlimit 1",
        " options ignore-unknown-layer-labels",
        "",
    ]
    # conductor layer defs: magic type <- CIF layer name; metals also read labels.
    # Contact/via SOURCE tokens are NOT conductors -- they are painted as contact
    # types below, so skip them here.
    for tok in layers:
        if tok in _CONTACT_SOURCE:
            continue
        mtype = _TOKEN_PLANE[tok][0]
        name = tok.upper()
        lines.append(f" layer {mtype} {name}")
        if tok in ("metal1", "metal2"):
            lines.append(f"  labels {name}TXT")
    # contact type defs: paint the contact type from its GDS contact/via layer,
    # AND'ed (when needed) against the underlying conductor to split poly vs diff.
    for c in _present_contacts(layers):
        lines.append(f" layer {c.name} {c.src_token.upper()}")
        if c.and_token is not None:
            lines.append(f"  and {c.and_token.upper()}")
    lines.append("")
    # calma binding block: CIF name -> (gds_layer, gds_datatype)
    for tok, (gl, gd) in layers.items():
        name = tok.upper()
        lines.append(f" calma {name} {gl} {gd}")
        if tok in ("metal1", "metal2"):
            lines.append(f" calma {name}TXT {gl} {_LABEL_TEXTTYPE}")
    lines.append("end")
    return "\n".join(lines)


def _present_contacts(
    layers: Dict[str, Tuple[int, int]]
) -> List[_ContactSpec]:
    """The contact specs whose source layer + both endpoint conductors resolved."""
    mtypes = {_TOKEN_PLANE[t][0] for t in layers if t not in _CONTACT_SOURCE}
    out: List[_ContactSpec] = []
    for c in _CONTACTS:
        if (
            c.src_token in layers
            and (c.and_token is None or c.and_token in layers)
            and c.lower in mtypes
            and c.upper in mtypes
        ):
            out.append(c)
    return out


def _emit_cifoutput(layers: Dict[str, Tuple[int, int]]) -> str:
    """Minimal ``cifoutput`` so Magic can round-trip the same layers to GDS."""
    lines = [
        "cifoutput",
        "style gdsii",
        " scalefactor 1 nanometers",
        " gridlimit 1",
    ]
    for tok, (gl, gd) in layers.items():
        if tok in _CONTACT_SOURCE:
            continue  # contact/via1 are not conductors; nothing to round-trip
        mtype = _TOKEN_PLANE[tok][0]
        name = tok.upper()
        lines.append("")
        lines.append(f" layer {name} {mtype}")
        lines.append(f"\tcalma {gl} {gd}")
    lines.append("end")
    return "\n".join(lines)


def _emit_styles(layers: Dict[str, Tuple[int, int]]) -> str:
    """Minimal display ``styles`` section (required before ``compose``).

    Each magic conductor AND each contact type gets a display style; the values
    are cosmetic (display only) and do not affect extraction, but every type that
    appears in ``types`` / ``contact`` must have a style or Magic flags an error.
    """
    # cosmetic display styles only; pick names known to exist in styletype mos.
    style_for = {
        "nwell": "nwell", "ndiff": "ndiffusion", "poly": "polysilicon",
        "metal1": "metal1", "metal2": "metal2",
        "pcont": "metal1", "dcont": "metal1", "m1via": "metal2",
    }
    lines = ["styles", " styletype mos"]
    seen: set = set()
    for tok in layers:
        if tok in _CONTACT_SOURCE:
            continue
        mtype = _TOKEN_PLANE[tok][0]
        if mtype in seen:
            continue
        seen.add(mtype)
        lines.append(f"  {mtype} {style_for.get(mtype, mtype)}")
    for c in _present_contacts(layers):
        if c.name in seen:
            continue
        seen.add(c.name)
        lines.append(f"  {c.name} {style_for.get(c.name, 'metal1')}")
    lines.append("end")
    return "\n".join(lines)


def _emit_compose() -> str:
    """Empty ``compose`` section (no composite/stacked types in this minimal tech)."""
    return "\n".join(["compose", "end"])


def _emit_connect(layers: Dict[str, Tuple[int, int]]) -> str:
    """``connect`` section: declares which types are electrically connected.

    Required before ``extract``.  This is where Magic's net merging is wired:

      * each metal plane connects to itself AND to every contact RESIDUE that
        lands on it (``pcont/m1``, ``dcont/m1``, ``m1via/m1`` on metal1;
        ``m1via/m2`` on metal2), so a metal fragment merges with the contact and,
        through the contact's other plane, with the conductor below;
      * poly connects to itself + its poly-contact active residue (``pcont/a``);
        ndiff connects to itself + its diff-contact active residue (``dcont/a``).
        Because poly and diff are listed in SEPARATE connect groups, they merge
        only where a contact bridges them -- never blanket-plane.

    ``<type>/<planeabbr>`` names the residue of a contact type on a given plane.
    """
    contacts = _present_contacts(layers)
    # residues that land on the metal1 plane: poly/diff contacts (upper=metal1)
    # AND the metal1->metal2 via (lower=metal1).
    m1_res = [f"{c.name}/m1" for c in contacts
              if c.upper == "metal1" or c.lower == "metal1"]
    m2_res = [f"{c.name}/m2" for c in contacts if c.upper == "metal2"]
    # active-plane residues of the contacts whose lower conductor is poly / ndiff
    poly_res = [f"{c.name}/a" for c in contacts if c.lower == "poly"]
    diff_res = [f"{c.name}/a" for c in contacts if c.lower == "ndiff"]

    def _grp(parts: List[str]) -> str:
        return ",".join(parts)

    lines = ["connect"]
    if "metal1" in layers:
        g = _grp(["metal1", *dict.fromkeys(m1_res)])
        lines.append(f"  {g} {g}")
    if "metal2" in layers:
        g = _grp(["metal2", *dict.fromkeys(m2_res)])
        lines.append(f"  {g} {g}")
    if "poly" in layers:
        g = _grp(["poly", *dict.fromkeys(poly_res)])
        lines.append(f"  {g} {g}")
    if "diff" in layers:
        g = _grp(["ndiff", *dict.fromkeys(diff_res)])
        lines.append(f"  {g} {g}")
    lines.append("end")
    return "\n".join(lines)


def _emit_drc() -> str:
    """Empty ``drc`` section (extraction needs the header present, no rules)."""
    return "\n".join(["drc", "end"])


def _emit_extract(
    layers: Dict[str, Tuple[int, int]], coeffs: Dict[str, float]
) -> str:
    """``extract`` section: planeorder + the cap coefficient lines.

    Emits ``defaultareacap`` / ``defaultperimeter`` / ``defaultsidewall`` for
    each conductor present, plus ``defaultoverlap`` / ``defaultsideoverlap`` for
    the metal2-over-metal1 pair when both metals resolved. All values are the
    per-PDK geometric coefficients from :func:`_cap_coeffs`.
    """
    area = _fmt(coeffs["areacap"])
    perim = _fmt(coeffs["perimcap"])
    side = _fmt(coeffs["sidewall"])
    over = _fmt(coeffs["overlap"])
    sideover = _fmt(coeffs["sideoverlap"])

    lines: List[str] = [
        "extract",
        " style ngspice",
        " cscale 1",
        " lambda 1.0",
        " units microns",
        " step 4",
        " sidehalo 8",
        " fringeshieldhalo 8",
        "",
    ]
    # planeorder for every declared plane (bottom -> top)
    for i, (plane, _abbr) in enumerate(_PLANES):
        lines.append(f" planeorder {plane} {i}")
    lines.append("")

    # per-conductor self caps (to substrate / lower plane)
    if "poly" in layers:
        lines.append(f" defaultareacap   poly active {area}")
        lines.append(f" defaultperimeter poly active {perim}")
        lines.append(f" defaultsidewall  poly active {side}")
        lines.append("")
    if "metal1" in layers:
        lines.append(f" defaultareacap   metal1 metal1 {area}")
        lines.append(f" defaultperimeter metal1 metal1 {perim}")
        lines.append(f" defaultsidewall  metal1 metal1 {side}")
        lines.append("")
    if "metal2" in layers:
        lines.append(f" defaultareacap   metal2 metal2 {area}")
        lines.append(f" defaultperimeter metal2 metal2 {perim}")
        lines.append(f" defaultsidewall  metal2 metal2 {side}")
        lines.append("")
    # metal-over-metal coupling: metal2 over metal1
    if "metal1" in layers and "metal2" in layers:
        lines.append(f" defaultoverlap     metal2 metal2 metal1 metal1 {over}")
        lines.append(f" defaultsideoverlap metal2 metal2 metal1 metal1 {sideover}")
        lines.append(f" defaultsideoverlap metal1 metal1 metal2 metal2 {sideover}")
        lines.append("")
    # metal1 over poly
    if "metal1" in layers and "poly" in layers:
        lines.append(f" defaultoverlap     metal1 metal1 poly active {over}")
        lines.append(f" defaultsideoverlap metal1 metal1 poly active {sideover}")
        lines.append("")
    lines.append("end")
    return "\n".join(lines)


def _emit_types(layers: Dict[str, Tuple[int, int]]) -> str:
    """Declare each magic conductor type AND each contact type on its home plane.

    A contact type's home plane is its UPPER plane (metal1 for poly/diff
    contacts, metal2 for the metal1->metal2 via); Magic auto-creates its lower
    residue from the ``contact`` section.
    """
    lines = ["types"]
    seen: set = set()
    for tok in layers:
        if tok in _CONTACT_SOURCE:
            continue
        mtype, plane = _TOKEN_PLANE[tok]
        key = (plane, mtype)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {plane} {mtype}")
    # contact types live on their UPPER plane
    upper_plane = {"metal1": "metal1", "metal2": "metal2"}
    for c in _present_contacts(layers):
        plane = upper_plane.get(c.upper, c.upper)
        key = (plane, c.name)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {plane} {c.name}")
    lines.append("end")
    return "\n".join(lines)


def _emit_contacts(layers: Dict[str, Tuple[int, int]]) -> str:
    """``contact`` section: declare each first-class Magic contact type.

    ``contact <name> <lower-type> <upper-type>`` tells Magic the type bridges the
    two planes those types live on, so the extractor forms a vertical net
    connection through it.  ``pcont poly metal1`` / ``dcont ndiff metal1`` /
    ``m1via metal1 metal2`` -- the poly/diff/metal types must be plain conductor
    TYPES (not plane names), or Magic rejects the section.
    """
    lines = ["contact"]
    for c in _present_contacts(layers):
        lines.append(f"  {c.name} {c.lower} {c.upper}")
    lines.append("end")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def magic_tech_from_pdk(
    pdk_id: str, *, root: Optional[str] = None
) -> GeneratedTech:
    """Generate the minimal Magic ``.tech`` for ``pdk_id`` from the PDK's own data.

    Resolves the real drawn-layer GDS numbers (``parse_layermap``) and the real
    per-PDK BEOL constants (``resolve_pdk_pex_constants``), derives the Magic cap
    coefficients, and assembles the tech sections. Emits only layers the PDK's
    layermap actually resolved. Returns a :class:`GeneratedTech` carrying the
    text + the per-coefficient provenance.

    Raises ``ValueError`` if NO routing layers (metal1/poly/contact) resolve for
    the PDK -- there is then nothing Magic could extract.
    """
    # 1) real GDS layer numbers, traced to the PDK's own deck / .lyp.
    # Conductors (drawn on planes) come from _TOKEN_PLANE; the contact/via1
    # SOURCE layers (which carry no plane of their own -- they are painted as
    # first-class Magic contact types) come from _CONTACT_SOURCE.
    detail = parse_layermap_detail(_pdk_root(pdk_id, root))
    known = set(_TOKEN_PLANE) | _CONTACT_SOURCE
    raw_layers: Dict[str, Tuple[int, int]] = {
        tok: tuple(info["gds"])  # type: ignore[arg-type]
        for tok, info in detail["layers"].items()  # type: ignore[union-attr]
        if tok in known
    }
    # preserve a stable emission order (bottom -> top of the stack)
    order = ["nwell", "diff", "poly", "contact", "metal1", "via1", "metal2"]
    layers = {t: raw_layers[t] for t in order if t in raw_layers}

    if not ({"metal1", "poly", "contact"} & set(layers)):
        raise ValueError(
            f"{pdk_id}: layermap resolved no routing layer (metal1/poly/contact); "
            f"resolved={sorted(layers)} from {detail.get('source_file')!r} -- "
            "nothing for Magic to extract"
        )

    # 2) real per-PDK BEOL constants -> Magic cap coefficients
    consts = resolve_pdk_pex_constants(pdk_id, root=root)
    coeffs = _cap_coeffs(consts)

    # 3) assemble the tech text
    header = (
        "# Auto-generated minimal Magic techfile for parasitic extraction.\n"
        f"# PDK: {pdk_id}\n"
        "# Generated by clay.layout.magic_tech_gen from the PDK's OWN data:\n"
        "#   cifinput GDS numbers  <- clay.frontend.pdk_layermap.parse_layermap\n"
        "#   cap coefficients      <- clay.layout.pex.pdk_constants."
        "resolve_pdk_pex_constants\n"
        "# This is a Magic 2.5D extraction from a Clay-generated tech with REAL\n"
        "# per-PDK constants; NOT the foundry signed StarRC/QRC ITF deck.\n"
    )
    tech_section = "\n".join(["tech", " format 35", f" {pdk_id}_clay", "end"])
    planes_section = "\n".join(
        ["planes"] + [f"  {name},{abbr}" for name, abbr in _PLANES] + ["end"]
    )
    # Magic's required section ORDER: tech, planes, types, contact, styles,
    # compose, connect, cifoutput, cifinput, drc, extract. (extract has a
    # prerequisite on connect; cif* must follow compose.)
    parts = [
        header,
        tech_section,
        "",
        planes_section,
        "",
        _emit_types(layers),
        "",
        _emit_contacts(layers),
        "",
        _emit_styles(layers),
        "",
        _emit_compose(),
        "",
        _emit_connect(layers),
        "",
        _emit_cifoutput(layers),
        "",
        _emit_cifinput(layers),
        "",
        _emit_drc(),
        "",
        _emit_extract(layers, coeffs),
        "",
    ]
    text = "\n".join(parts)

    return GeneratedTech(
        pdk_id=pdk_id,
        text=text,
        layers=layers,
        coeffs=coeffs,
        constants=consts,
        layermap_source=detail.get("source_file"),  # type: ignore[arg-type]
    )


def _pdk_root(pdk_id: str, root: Optional[str]) -> str:
    if root:
        return root
    from clay.layout.pex.pdk_constants import _resolve_root

    r = _resolve_root(pdk_id, None)
    if r is None:
        raise ValueError(f"cannot resolve PDK root for {pdk_id!r}")
    return str(r)


def tech_provenance(gt: GeneratedTech) -> Dict[str, object]:
    """Trace every cap coefficient back to the per-PDK constant that produced it.

    Returns a dict suitable for logging / asserting: each coefficient name maps
    to its value, the formula, and the source constant(s) (with origin pdk/default).
    """
    c = gt.constants
    prov = c.provenance

    def src(field: str) -> Dict[str, object]:
        p = prov.get(field, {})
        return {
            "value": p.get("value"),
            "origin": p.get("origin"),
            "source": p.get("source") or p.get("note"),
            "datum": p.get("datum"),
        }

    return {
        "pdk_id": gt.pdk_id,
        "layermap_source": gt.layermap_source,
        "layers": {k: list(v) for k, v in gt.layers.items()},
        "constants": {
            "eps_ild_aF_per_um": c.eps_ild_aF_per_um,
            "metal_thickness_um": src("metal_thickness_um"),
            "ild_k": src("ild_k"),
            "metal_vgap_um": src("metal_vgap_um"),
            "min_space_um": src("min_space_um"),
        },
        "coefficients": {
            "areacap_aF_per_um2": {
                "value": gt.coeffs["areacap"],
                "formula": "eps_ild / metal_vgap_um",
            },
            "sidewall_aF_per_um": {
                "value": gt.coeffs["sidewall"],
                "formula": "eps_ild * (metal_thickness_um / min_space_um) * k_fringe",
                "k_fringe": _K_FRINGE,
            },
        },
        "calibration": (
            "Magic 2.5D from a Clay-generated tech (real per-PDK constants); "
            "NOT foundry StarRC/QRC ITF; NOT a field solve. calibrated=False."
        ),
    }


def write_tech(gt: GeneratedTech, out_dir: str) -> str:
    """Write the generated tech to ``<out_dir>/<pdk_id>_clay.tech``; return path."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{gt.pdk_id}_clay.tech"
    path.write_text(gt.text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Magic run (Docker), mirroring clay.layout.pex.magic_runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MagicRunResult:
    loaded: bool                 # did Magic load the tech + GDS without fatal error
    cap_count: int               # number of C<n> cap lines in the ext2spice output
    spice_text: str              # the ext2spice output (empty if it failed)
    returncode: int
    stdout: str
    stderr: str
    error: Optional[str] = None  # exact Magic error if it rejected the tech


def _count_caps(spice_text: str) -> int:
    n = 0
    for line in spice_text.splitlines():
        s = line.strip()
        if s[:1].upper() == "C" and len(s) > 1 and (s[1].isdigit() or s[1].isalpha()):
            # an ext2spice cap line: "C0 net1 net2 1.23fF"
            parts = s.split()
            if len(parts) >= 3:
                n += 1
    return n


def run_magic_pex(
    gds_path: str,
    tech_path: str,
    cell_name: str,
    *,
    image: str = "hpretl/iic-osic-tools:latest",
    timeout: int = 600,
) -> MagicRunResult:
    """Run Magic ext2spice in Docker with the GENERATED tech on a real GDS.

    Mirrors :func:`clay.layout.pex.magic_runner.run_ext2spice` but loads the
    generated ``.tech`` via ``-T`` instead of a PDK ``.magicrc``. Bind-mounts the
    GDS directory (and copies the tech into it) as ``/work``. Returns whether
    Magic loaded the tech cleanly and the extracted cap count, or the exact error.
    """
    import shutil
    import subprocess

    docker = shutil.which("docker")
    if not docker:
        return MagicRunResult(
            loaded=False, cap_count=0, spice_text="", returncode=-1,
            stdout="", stderr="", error="docker CLI not found",
        )

    gds = Path(gds_path).resolve()
    workdir = gds.parent
    tech = Path(tech_path).resolve()
    # stage the tech into the mounted dir so the container can read it
    staged_tech = workdir / tech.name
    if staged_tech.resolve() != tech:
        staged_tech.write_text(tech.read_text(encoding="utf-8"), encoding="utf-8")
    mount = str(workdir).replace("\\", "/")

    out_spice = f"{cell_name}.clay_pex.spice"
    # Force OUR generated tech explicitly (the container's PDK .magicrc may have
    # auto-loaded a different tech at startup; `tech load -noprompt` overrides it).
    tcl = (
        f"tech load /work/{tech.name} -noprompt\n"
        f"gds read /work/{gds.name}\n"
        f"load {cell_name}\n"
        "select top cell\n"
        "extract all\n"
        "ext2spice lvs\n"
        "ext2spice cthresh 0\n"
        "ext2spice rthresh 0\n"
        f"ext2spice -o /work/{out_spice}\n"
        "quit -noprompt\n"
    )
    tcl_name = "_clay_gentech_extract.tcl"
    (workdir / tcl_name).write_text(tcl, encoding="utf-8")

    # -noconsole/-dnull batch; -norcfile so no PDK .magicrc forces a foreign tech.
    magic_cmd = (
        f"magic -dnull -noconsole -norcfile -T /work/{tech.name} /work/{tcl_name}"
    )
    proc = subprocess.run(
        [docker, "run", "--rm", "-v", f"{mount}:/work",
         "--entrypoint", "bash", image, "-lc", magic_cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    out_file = workdir / out_spice
    spice_text = out_file.read_text() if out_file.exists() else ""

    # detect a tech-load rejection in Magic's stderr/stdout
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    tech_error = None
    for marker in ("Couldn't find a technology",
                   "is not a valid technology",
                   "Error in technology file",
                   "Failure to read entire technology",
                   "Error loading technology",
                   "appears too early",
                   "Unrecognized layer",
                   "no such technology"):
        if marker.lower() in combined.lower():
            tech_error = marker
            break

    # The ext2spice header cites the technology actually used. Our tech is loaded
    # ONLY if that line names ``<pdk>_clay`` -- otherwise a startup .magicrc tech
    # (e.g. the image's sg13g2) produced the output and OUR tech was rejected.
    tech_name = f"{Path(tech_path).stem}"
    used_our_tech = (
        f"technology: {tech_name}" in spice_text
        or f"technology: {tech_name.replace('_clay', '')}_clay" in spice_text
    )
    loaded = out_file.exists() and tech_error is None and used_our_tech
    err = None
    if not out_file.exists():
        err = tech_error or f"no ext2spice output (rc={proc.returncode})"
    elif tech_error is not None:
        err = tech_error
    elif not used_our_tech:
        head = spice_text.splitlines()[0] if spice_text else "(empty)"
        err = (
            "ext2spice ran under a DIFFERENT technology (startup .magicrc), not "
            f"the generated tech; output header: {head!r}"
        )

    return MagicRunResult(
        loaded=loaded,
        cap_count=_count_caps(spice_text),
        spice_text=spice_text,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        error=err,
    )


def magic_tech_loads_clean(
    tech_path: str,
    *,
    image: str = "hpretl/iic-osic-tools:latest",
    timeout: int = 300,
) -> Tuple[bool, str]:
    """Probe whether Magic loads ``tech_path`` cleanly (no GDS needed).

    Runs ``tech load <tech>`` in a batch Magic and checks that ``tech name``
    reports the generated tech AND no section/calma/residue error appeared.
    Returns ``(ok, detail)`` where ``detail`` is the loaded tech name on success
    or the first Magic error on failure. Returns ``(False, "no docker")`` when
    Docker is unavailable (gates out like the rest of the Magic path).
    """
    import shutil
    import subprocess

    docker = shutil.which("docker")
    if not docker:
        return False, "docker CLI not found"
    tech = Path(tech_path).resolve()
    workdir = tech.parent
    mount = str(workdir).replace("\\", "/")
    tcl_name = "_clay_techprobe.tcl"
    (workdir / tcl_name).write_text(
        f"tech load /work/{tech.name} -noprompt\n"
        'puts "CLAY_TECHNAME: [tech name]"\n'
        "quit -noprompt\n",
        encoding="utf-8",
    )
    cmd = f"magic -dnull -noconsole -norcfile /work/{tcl_name}"
    proc = subprocess.run(
        [docker, "run", "--rm", "-v", f"{mount}:/work",
         "--entrypoint", "bash", image, "-lc", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    expect = tech.stem
    name_ok = f"CLAY_TECHNAME: {expect}" in combined
    err = None
    for marker in ("Unrecognized layer", "missing a residue",
                   "Wrong number of arguments", "Error in technology file",
                   "appears too early", "Error loading technology",
                   "Missing prerequisite", "contained errors"):
        if marker.lower() in combined.lower():
            err = marker
            break
    if name_ok and err is None:
        return True, expect
    return False, err or f"tech name not reported as {expect!r}"


__all__ = [
    "GeneratedTech",
    "MagicRunResult",
    "magic_tech_from_pdk",
    "tech_provenance",
    "write_tech",
    "run_magic_pex",
    "magic_tech_loads_clean",
]
