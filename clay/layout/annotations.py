from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import List, Optional

from .symbolic_layout import SymbolicLayout


class WellType(Enum):
    NWELL = "nwell"
    PWELL = "pwell"


class MatchingKind(Enum):
    RANDOM_VT = "random_vt"
    SYSTEMATIC = "systematic"
    BOTH = "both"


class SymmetryAxis(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class InterdigitationPattern(Enum):
    ABAB = "abab"
    ABBA = "abba"


class Orientation(Enum):
    R0 = "R0"
    R90 = "R90"
    R180 = "R180"
    R270 = "R270"
    MR0 = "MR0"
    MR90 = "MR90"
    MR180 = "MR180"
    MR270 = "MR270"


class LayoutAnnotation:
    """Base class; concrete annotations subclass below."""
    pass


@dataclass(eq=True, frozen=True)
class NativeBodyTap(LayoutAnnotation):
    """Select the native PCell body contact layout for a device."""
    mode: str = "single"


@dataclass(eq=True)
class Matching(LayoutAnnotation):
    other: str
    kind: MatchingKind = MatchingKind.RANDOM_VT


@dataclass(eq=True)
class GuardRing(LayoutAnnotation):
    well_type: WellType


@dataclass(eq=True)
class WellIsolated(LayoutAnnotation):
    from_device: str


@dataclass(eq=True)
class Symmetric(LayoutAnnotation):
    devices: List[str]
    axis: SymmetryAxis = SymmetryAxis.VERTICAL


@dataclass(eq=True)
class Interdigitated(LayoutAnnotation):
    with_device: str
    pattern: InterdigitationPattern = InterdigitationPattern.ABAB
    finger_pairs: int = 1


@dataclass(eq=True)
class Orient(LayoutAnnotation):
    """Declared orthogonal device transform (``R*`` rotation or ``MR*`` mirror).

    R0 is the default; R90/R270 rotate by a quarter turn and R180 flips it.
    ``MR*`` applies the corresponding coordinate-free orthogonal reflection.
    The physical terminal boxes transform with the artwork while their logical
    net bindings remain unchanged.  Individual builders fail closed when a
    transform has not been proved for that device family.
    """
    orientation: Orientation = Orientation.R0


@dataclass(eq=True)
class DenseContacts(LayoutAnnotation):
    """Declared REFERENCE-PARITY contact density (`@dense_contacts`).

    Stamps a FULL source/drain contact-cut column (``build_device(contact_rows=
    True)``) for this single-row MOS instead of the sparse ~1-cut default -- the
    external_pdk reference packs ~42 cuts per S/D face (cont=88 on a W=21 finger) where the
    sparse column leaves ~1-4. A GEOMETRIC device property, LVS-invariant (more cuts
    on the SAME net). Single-row only (contact_rows is unsupported with a serpentine
    ``fold_rows>1`` build, so a serpentine-folded device silently keeps the sparse
    column). Carried as a device-level annotation (like `@orient`) so the intent is
    DECLARED in the .clay, not a hidden build flag; the geometric_fidelity
    ``contact_density`` metric verifies the emitted column."""
    pass


@dataclass(eq=True)
class Fold(LayoutAnnotation):
    """Declared DEVICE FOLDING into bounded-height multi-finger fingers (`@fold`).

    A wide/tall MOS that the netlist draws as a SINGLE tall finger (e.g. the OTA
    bias NMOS ``MM15``/``MM17`` at ``W=40u nf(m)=1`` -- a ~42um-tall diffusion
    strip) is instead laid out as ``F`` physical fingers EACH ``W_eff / F`` tall,
    with ``F`` chosen so no finger exceeds ``h_max_um`` (the reference's compact
    unit-transistor style). The device is ELECTRICALLY IDENTICAL -- the total gate
    width ``W_eff = nf * W`` is conserved EXACTLY (``F * (W_eff/F) == W_eff``); only
    the per-finger HEIGHT changes (``W=40 nf=1`` -> ``F=2`` fingers of 20um, halving
    the strip height). A device whose fingers are already <= ``h_max`` is unchanged.

    The finger plan is the SHARED :func:`clay.layout.extract.fold.folded_finger_plan`
    (``W_eff = nf*W``; ``F = max(nf, ceil(W_eff/h_max))``; ``W_finger = W_eff/F``) --
    the SAME arithmetic the migrate path uses, so the placer footprint, the route
    stamp, and the emitted GDS all fold identically (one canonical geometry). It is
    a GEOMETRIC device property, LVS-invariant (same device, more fingers on the same
    nets). Works for ANY device / PDK that declares it -- no per-circuit branch.

    ``h_max_um`` is the maximum finger height (um). ``None`` uses the module default
    (:data:`clay.layout.extract.fold.DEFAULT_H_MAX_UM`, the reference's ~22um bound);
    an explicit ``@fold(<h_max>)`` overrides it. Carried as a device-level annotation
    (like ``@dense_contacts`` / ``@orient``) so the intent is DECLARED in the .clay,
    not a hidden build flag; an independent checker verifies the emitted device is
    multi-finger with bounded finger height and conserved ``W_eff``."""
    h_max_um: Optional[float] = None


@dataclass(eq=True, frozen=True)
class ResistorSegments(LayoutAnnotation):
    """PDK-bound straight-segment resistor topology.

    ``count`` straight resistor bodies of ``width_um`` by ``length_um`` are
    connected by metal in ``series`` or ``parallel``.  The declaration carries
    the source PCell's per-segment resistance so the physical consumer can prove
    that the topology realizes the schematic value before it draws artwork.

    This is deliberately opt-in.  A resistor without this annotation keeps the
    legacy value-driven constructor, which protects circuits whose source never
    declared a segmented PCell topology.
    """

    pdk: str
    count: int
    connection: str
    width_um: float
    length_um: float
    segment_resistance_ohms: float
    value_quantum_ohms: float
    # Zero preserves legacy declarations whose total-value interval already
    # proves equivalence.  A positive value carries the source segR LSB so the
    # consumer can intersect both independently rounded source intervals.
    segment_resistance_quantum_ohms: float = 0.0


@dataclass(eq=True)
class Adjacent(LayoutAnnotation):
    to: str
    max_separation_multiplier: Fraction = field(default_factory=lambda: Fraction(2, 1))


@dataclass(eq=True)
class EqualSize(LayoutAnnotation):
    """Declared EQUAL-FOOTPRINT group for a set of devices/passives
    (`@equal_size(d1, d2, ...)`, >=2 labels).

    The members must be placed at an IDENTICAL footprint — the same placed
    WIDTH and the same placed HEIGHT, within the placer's grid epsilon. This
    PINS the ``size_class`` invariant the layout language otherwise leaves
    implicit: two passives can end up equal-footprint only because a designer
    happened to hand-set both their schematic values equal (e.g. two 1.1p
    Millers/loads), with NOTHING in the layout language pinning the equality.
    ``@equal_size`` makes the equality a FIRST-CLASS declared intent the placer
    must realise and a ``check_equal_size`` obligation verifies on the real
    placement (relative footprint spread within the group <= a small rel tol).

    Carried as a (`@cell`-targeted) device-level annotation — a sibling of
    ``@symmetric`` — so the lark/pest parity lock stays closed and the
    translate path reads it off the same ``AnnotatedCell.annotations`` map.
    ``devices`` is the ordered tuple of member LABELS (devices or passives).
    """
    devices: List[str]


@dataclass(eq=True)
class Fix(LayoutAnnotation):
    """ABSOLUTE-PLACEMENT pin for one device (`@fix(device, x=<um>, y=<um>)`).

    The named ``device`` is placed with its CENTROID at the absolute position
    ``(x, y)`` (in microns), OVERRIDING whatever position the placer's engine
    would otherwise compute. ``x`` / ``y`` are the device centroid; the device's
    own built geometry (its width/height/orientation, from its cell) is preserved
    -- only its translation is fixed. This is the GROUND TERM of the intent-
    extraction method's bottom clause: when EVERY device in a cell carries a
    ``@fix``, the placer has ZERO remaining freedom, so any faithful placer (milp
    or seqpair) reproduces the SAME placement -- strict consistency by
    construction (``incompleteness_score == 0``).

    Carried as a (``@cell``-targeted) device-level annotation -- a sibling of
    ``@equal_size`` -- so the lark/pest parity lock stays closed and the translate
    path reads it off the same ``AnnotatedCell.annotations`` map. ``device`` is the
    pinned device LABEL; ``x`` / ``y`` are floats in microns (the device centroid).
    A ``check_fix`` obligation verifies the realised centroid lands at the pin on
    the real placement; an empty pin set leaves every placer byte-unchanged.
    """
    device: str
    x: float
    y: float


@dataclass(eq=True)
class Bulk(LayoutAnnotation):
    """Declared body/bulk net for a device (`@bulk(net)`).

    The body terminal of a MOS is the well/substrate tie. Schematics often
    leave the body implicit (tied to the source rail) or carry a generic
    `b` pin whose net is a placeholder. `@bulk(net)` is the DECLARED
    authority for which net the body actually connects to — it overrides
    any inferred/placeholder body net when a LayoutPlan reconciles bulk
    connectivity. Carrying it as a device-level annotation (not a new
    top-level construct) keeps the lark/pest parity lock closed.
    """
    net: str


@dataclass(eq=True)
class Tap(LayoutAnnotation):
    """Declared substrate/well tap for a device (`@tap(net, where=...)`).

    `net` is the tap net (the rail the tap ties to, e.g. `vss` for a
    p-substrate tie or `vdd` for an n-well tie). `where` is the LOCATION
    intent for the tap relative to the device — a bare identifier such as
    `substrate`, `ring`, `abut`, or `shared`. It is informational for the
    placer/tap generator; the LayoutPlan records it so a tap inserter can
    honour the declared placement instead of guessing.
    """
    net: str
    where: str


@dataclass(eq=True)
class AnnotatedCell:
    cell_name: str
    layouts: dict = field(default_factory=dict)        # dict[str, SymbolicLayout]
    annotations: dict = field(default_factory=dict)    # dict[str, list[LayoutAnnotation]]

    def set_layout(self, device: str, layout: SymbolicLayout) -> None:
        self.layouts[device] = layout

    def add_annotation(self, device: str, ann: LayoutAnnotation) -> None:
        self.annotations.setdefault(device, []).append(ann)

    def annotations_for(self, device: str) -> list:
        return list(self.annotations.get(device, []))

    def has_device(self, device: str) -> bool:
        return device in self.layouts
