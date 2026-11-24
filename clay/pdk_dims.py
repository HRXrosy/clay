"""
PdkDims — sizing-side process dimensional context (R95 E / M-PDK-DECOUPLE).

This module is the *sizing-side* complement to ``clay.layout.pdk_dims`` (which
exposes layout-side geometric pitches + Pelgrom A_VT mismatch coefficients).
Where layout-side ``PdkDims`` answers "what are the metal/poly/fin pitches and
the mismatch slope?", THIS sizing-side ``PdkDims`` answers a different
question:

    "When I want to express a transistor W as a multiple of 'PDK-natural
    minimum', what number do I multiply?"

Concretely, this module surfaces six per-PDK process constants that let .clay
sizing blocks be expressed in PDK units rather than raw SI metres:

  * ``min_W``      — minimum drawn width                    (metres)
  * ``min_L``      — minimum drawn channel length           (metres)
  * ``default_W``  — recommended starting W for analog      (metres)
  * ``default_L``  — recommended starting L for analog      (metres)
  * ``m_unit``     — multiplicity unit (1 = unit transistor)
  * ``pdk_name``   — PDK identifier string (e.g. ``"sky130"``)

The class is paired with a context-manager pattern so .clay author / sizing
script can write::

    with PdkDims.sky130() as dims:
        # W is now in PDK-natural units; raw SI value is computed lazily
        W = dims.min_W * 3              # 3× the minimum allowed width
        L = dims.default_L * 2          # 2× the recommended analog length

Why a separate ``clay.pdk_dims`` module (not folded into ``clay.pdk``)
-------------------------------------------------------------------
The ``clay.pdk`` adapter registry (``PdkAdapter``) lives at a *driver*
abstraction layer: it owns device-class bindings, μ·C_ox lookups, corner
file resolvers, etc. — the things drivers (ngspice / spectre / hspice) need
in order to emit a correct netlist.

``PdkDims`` here lives at a *sizing-author* abstraction layer: it owns the
process constants that .clay authors care about when writing sizing { } blocks
and want their W/L expressions to be PDK-agnostic.

These are distinct concerns:

  * The driver doesn't care that 0.42 µm is "3 × sky130 min_W"; it just
    needs the SI metres value to emit ``W=4.2e-7``.
  * The sizing author cares that the same expression "3 × min_W" should mean
    0.42 µm on sky130 but 0.66 µm on gf180mcu — without rewriting the .clay
    source per PDK.

Keeping the two surfaces in separate modules lets the driver layer evolve
(more corner names, lazy LUT loading, etc.) independently from the sizing-
author API.

R76 B / R77 B alignment
-----------------------
The numeric ``min_W`` / ``min_L`` values per PDK in this module are sourced
from the same documents as the R76 B (sky130) / R77 B (gf180mcu) gm/Id LUT
generator: ``tools/pdk_lut_generator/run_ngspice_sweep.py`` swept the same
grid, and the LUT JSON files (``data/pdk_luts/{sky130,gf180mcu}_gm_id.json``)
are calibrated to these ``min_W`` / ``min_L`` values. Sizing scripts that
mix LUT lookups with ``PdkDims`` therefore get a coherent process picture
(both surfaces draw from the same per-PDK source of truth).

The skeleton open_pdk / open_pdk entries are paper-grade defaults (no real
ngspice install / no real silicon validation) — consistent with the
``M-PDK-DECOUPLE`` open-foundry-only data closure scope. Their ``min_W`` /
``min_L`` values match the corresponding ``PdkAdapter`` skeletons in
``clay/pdk/open_pdk.py`` and ``clay/pdk/open_pdk.py``.

R95 E scope (what is and isn't here)
------------------------------------
This file lands the ``PdkDims`` context surface. Qualitative DRC annotation
(the other open arm of M-PDK-DECOUPLE) lives layout-side per
``CROSS_PROMPT_SCOPE.html`` §3.1 and is delegated to the layout prompt
(R013+ on its side). Wiring ``PdkDims`` into the sizing *cost function* (so
that the .clay author can spec a cost like ``area = W * L * m_unit``) is
deferred to R96+ — this file only lands the context surface itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping


# ──────────────────────────────────────────────────────────────────────
# Core dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PdkDims:
    """Sizing-side dimensional context for one PDK.

    All length values are in SI metres. Multiplicity unit (``m_unit``) is
    dimensionless: a unit transistor has ``m=1``, a finger-folded device
    has ``m=N`` for some integer N — sizing libraries use this as a knob
    independent of W/L.

    Use the factory classmethods (``PdkDims.sky130()``, ``.gf180mcu()``,
    ``.open_pdk()``, ``.open_pdk()``) rather than ``__init__`` directly so that
    the canonical per-PDK process constants are always pulled from one
    source of truth (the ``_DIM_REGISTRY`` table at the bottom of this
    module).

    Frozen on purpose: a ``PdkDims`` instance describes immutable process
    constants. Sizing scripts that want to vary W/L should multiply
    ``dims.min_W`` etc. — they should never mutate the ``PdkDims`` itself.
    """

    pdk_name: str          # PDK identifier (e.g. "sky130")
    min_W: float           # minimum drawn width (metres)
    min_L: float           # minimum drawn channel length (metres)
    default_W: float       # recommended analog starting W (metres)
    default_L: float       # recommended analog starting L (metres)
    m_unit: int = 1        # multiplicity unit (1 = unit transistor)

    # ─── Factory classmethods (one per PDK) ────────────────────────

    @classmethod
    def sky130(cls) -> "PdkDims":
        """Sky130 process dimensional context.

        Values calibrated against the R76 B gm/Id LUT generator inputs
        (``tools/pdk_lut_generator/run_ngspice_sweep.py``) and the
        ``sky130_fd_pr__nfet_01v8`` / ``__pfet_01v8`` device cards:

          * ``min_W = 0.42 µm`` — drawn min for the analog 1.8 V flavour
            (digital flavours go down to 0.36 µm but Clay defaults to
            analog).
          * ``min_L = 0.15 µm`` — drawn min channel length.
          * ``default_W = 1.0 µm`` — typical analog starting width.
          * ``default_L = 0.5 µm`` — typical analog starting length
            (matches ``clay/sizing/__init__.py::solve_sizing`` default).
        """
        return _DIM_REGISTRY["sky130"]

    @classmethod
    def gf180mcu(cls) -> "PdkDims":
        """GlobalFoundries gf180mcu 3.3 V process dimensional context.

        Values calibrated against the R77 B gm/Id LUT generator inputs and
        the ``nmos_3p3`` / ``pmos_3p3`` device cards in the sm141064
        release. ``min_W`` matches the ``extra["min_W"]`` field in the
        ``GF180MCU`` adapter.
        """
        return _DIM_REGISTRY["gf180mcu"]



    @classmethod
    def from_pdk_name(cls, pdk_name: str) -> "PdkDims":
        """Lookup ``PdkDims`` by PDK identifier string.

        Raises ``KeyError`` if the PDK is not recognised — symmetric with
        ``clay.pdk.get_pdk(pdk_id)`` behaviour.
        """
        if pdk_name not in _DIM_REGISTRY:
            raise KeyError(
                f"No PdkDims registered for {pdk_name!r}. "
                f"Known PDKs: {sorted(_DIM_REGISTRY)}."
            )
        return _DIM_REGISTRY[pdk_name]

    # ─── Context manager pattern ───────────────────────────────────

    def __enter__(self) -> "PdkDims":
        """Enter PdkDims-as-context-manager.

        Lets sizing scripts write::

            with PdkDims.sky130() as dims:
                W = dims.min_W * 3
                L = dims.default_L * 2

        The ``with`` block is a *scoping* convenience — no global state
        is mutated. It exists so that .clay authors who think "I am now
        in sky130-units mode" have a syntactic marker matching that
        mental model, and so that the same script can be re-targeted to
        a different PDK by swapping a single ``PdkDims.sky130()`` →
        ``PdkDims.gf180mcu()`` factory call.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context (no-op — ``PdkDims`` carries no scoped state).

        Returning ``None`` (falsy) lets any exception raised inside the
        ``with`` block propagate normally.
        """
        return None

    # ─── Convenience helpers ───────────────────────────────────────

    def scale_W(self, multiplier: float) -> float:
        """Return ``multiplier × min_W`` in metres.

        Sugar for the common case of expressing W as N× the minimum
        allowed width. ``multiplier = 1.0`` returns the unit-W device.
        """
        return multiplier * self.min_W

    def scale_L(self, multiplier: float) -> float:
        """Return ``multiplier × min_L`` in metres.

        Sugar for the common case of expressing L as N× the minimum
        allowed channel length.
        """
        return multiplier * self.min_L

    def scale_default_W(self, multiplier: float) -> float:
        """Return ``multiplier × default_W`` in metres.

        Use when the .clay author wants to express W as a multiple of
        the *recommended analog* starting width rather than the absolute
        minimum (the latter is often too small for analog mismatch).
        """
        return multiplier * self.default_W

    def scale_default_L(self, multiplier: float) -> float:
        """Return ``multiplier × default_L`` in metres."""
        return multiplier * self.default_L

    def is_compatible_W(self, W: float) -> bool:
        """True iff ``W >= min_W`` (i.e. width does not violate min DRC)."""
        return W >= self.min_W

    def is_compatible_L(self, L: float) -> bool:
        """True iff ``L >= min_L`` (i.e. length does not violate min DRC)."""
        return L >= self.min_L

    def with_m(self, m: int) -> "PdkDims":
        """Return a copy with a different multiplicity unit.

        Sizing scripts that want to express device folding as a context-
        scoped knob can write::

            with PdkDims.sky130().with_m(4) as dims:
                # All later W expressions implicitly mean "this W per
                # finger, with 4 fingers".
                W_per_finger = dims.min_W * 2
                W_total = W_per_finger * dims.m_unit  # = 8 × min_W

        Returns a new frozen instance — the original is untouched.
        """
        if m < 1:
            raise ValueError(f"m_unit must be >= 1, got {m}")
        return PdkDims(
            pdk_name=self.pdk_name,
            min_W=self.min_W,
            min_L=self.min_L,
            default_W=self.default_W,
            default_L=self.default_L,
            m_unit=m,
        )


# ──────────────────────────────────────────────────────────────────────
# Per-PDK process-constant registry (single source of truth)
# ──────────────────────────────────────────────────────────────────────

# Sky130: open-PDK sky130_fd_pr 1.8 V analog flavour.
#   min_W   = 0.42 µm  (sky130_fd_pr__nfet_01v8 / __pfet_01v8 drawn min)
#   min_L   = 0.15 µm  (matches clay/pdk/sky130.py::SKY130.min_L)
#   default = 1.0 µm × 0.5 µm  (matches sizing/__init__.py L_default)
_SKY130 = PdkDims(
    pdk_name="sky130",
    min_W=0.42e-6,
    min_L=0.15e-6,
    default_W=1.0e-6,
    default_L=0.5e-6,
    m_unit=1,
)

# Gf180mcu: open-PDK gf180mcu_fd_pr 3.3 V flavour (sm141064 release).
#   min_W   = 0.22 µm  (matches clay/pdk/gf180mcu.py::GF180MCU.extra["min_W"])
#   min_L   = 0.28 µm  (matches GF180MCU.min_L — 3p3 drawn min)
#   default = 1.0 µm × 0.5 µm  (typical 180 nm analog starting point)
_GF180MCU = PdkDims(
    pdk_name="gf180mcu",
    min_W=0.22e-6,
    min_L=0.28e-6,
    default_W=1.0e-6,
    default_L=0.5e-6,
    m_unit=1,
)

# open_pdk: predictive FinFET skeleton (no real ngspice install in CI).
#   min_W   = 27 nm    (matches clay/pdk/open_pdk.py::open_pdk.extra["min_W"])
#   min_L   = 7 nm     (matches open_pdk.min_L — 7 nm drawn min)
#   default = 100 nm × 7 nm  (deep-submicron analog starting cut)

# open_pdk: hypothetical 28 nm planar (paper-grade only — closed foundry).
#   min_W   = 80 nm    (matches clay/pdk/open_pdk.py::open_pdk.extra["min_W"])
#   min_L   = 28 nm    (matches open_pdk.min_L — 28 nm drawn min)
#   default = 200 nm × 30 nm  (28 nm-class analog starting cut)

_DIM_REGISTRY: Mapping[str, PdkDims] = {
    "sky130":   _SKY130,
    "gf180mcu": _GF180MCU,
}


# ──────────────────────────────────────────────────────────────────────
# Cross-PDK comparison helper
# ──────────────────────────────────────────────────────────────────────


def compare_min_W(pdk_a: str, pdk_b: str) -> float:
    """Return ``PdkDims(pdk_a).min_W / PdkDims(pdk_b).min_W``.

    Useful for sizing scripts that want to express "the same logical
    transistor on a different PDK" — multiply the W by this ratio to
    preserve the per-PDK min_W multiplier on the new PDK.

    Example::

        # I have W = 0.84 µm = 2 × min_W on sky130. What W gives the
        # same "2 × min_W" on gf180mcu?
        ratio = compare_min_W("gf180mcu", "sky130")  # 0.524
        W_gf = 0.84e-6 * ratio                       # 0.44 µm
        assert W_gf == 2 * PdkDims.gf180mcu().min_W

    Raises ``KeyError`` if either PDK is not recognised.
    """
    a = PdkDims.from_pdk_name(pdk_a)
    b = PdkDims.from_pdk_name(pdk_b)
    return a.min_W / b.min_W


def compare_min_L(pdk_a: str, pdk_b: str) -> float:
    """Return ``PdkDims(pdk_a).min_L / PdkDims(pdk_b).min_L``.

    Symmetric with :func:`compare_min_W` for channel length. Useful when
    a sizing recipe says "L = 5 × min_L" and the .clay author wants the
    SI metres value on a different process.
    """
    a = PdkDims.from_pdk_name(pdk_a)
    b = PdkDims.from_pdk_name(pdk_b)
    return a.min_L / b.min_L


# ──────────────────────────────────────────────────────────────────────
# Context manager helper for "swap PDK for one expression"
# ──────────────────────────────────────────────────────────────────────


@contextmanager
def using_pdk_dims(pdk_name: str) -> Iterator[PdkDims]:
    """Convenience context manager: ``with using_pdk_dims("sky130") as dims:``

    Equivalent to ``with PdkDims.from_pdk_name(pdk_name) as dims:``. Exists
    so sizing scripts can swap PDK at the top of a function via one symbol
    rather than going through the classmethod factory.

    Raises ``KeyError`` if ``pdk_name`` is not in the registry.
    """
    dims = PdkDims.from_pdk_name(pdk_name)
    with dims as scoped:
        yield scoped


# ──────────────────────────────────────────────────────────────────────
# Module-level discovery
# ──────────────────────────────────────────────────────────────────────


def list_known_pdks() -> list[str]:
    """Return sorted list of PDK names registered with ``PdkDims``.

    Currently: ``["open_pdk", "gf180mcu", "sky130", "open_pdk"]``. Third-party
    PDKs can register additional entries by calling :func:`register_dims`.
    """
    return sorted(_DIM_REGISTRY)


def register_dims(dims: PdkDims) -> None:
    """Register a third-party ``PdkDims`` instance.

    Symmetric with ``clay.pdk.register_pdk(adapter)`` — lets PDK-vendor
    packages plug in without modifying Clay's source. Re-registering an
    existing PDK identifier raises ``ValueError`` to prevent silent
    process-constant drift between two competing definitions.
    """
    # Mutate the underlying dict — _DIM_REGISTRY is typed as Mapping for
    # read-only callers but the registration path is internal-only.
    registry: dict[str, PdkDims] = _DIM_REGISTRY  # type: ignore[assignment]
    if dims.pdk_name in registry and registry[dims.pdk_name] is not dims:
        raise ValueError(
            f"PdkDims for {dims.pdk_name!r} is already registered. "
            f"Refusing to silently override existing constants. "
            f"If this is intentional, unregister first."
        )
    registry[dims.pdk_name] = dims


__all__ = [
    "PdkDims",
    "compare_min_W",
    "compare_min_L",
    "using_pdk_dims",
    "list_known_pdks",
    "register_dims",
]
