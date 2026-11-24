"""
PDK retarget — analytic gm/Id- + headroom-preserving sizing rescale.

Front-end paradigm-mapper, Agent C. Today a design is brought onto a new
PDK by HAND: someone re-derives W/L/m and the bias voltages for the new
supply, threshold and mobility-Cox product (see the "Sizing Derivation
Log" in ``circuits/fc_ota/DESIGN.md`` / the ``fc_ota.cir`` header). This
module turns that hand derivation into a general, reviewable engine.

Two layers:

  1. :func:`retarget_params` — the DEFAULT path. A closed-form square-law
     rescale that holds each device's operating point (gm/Id and Vov) while
     swapping the process constants. Needs no simulator and no optimiser.

  2. :func:`retarget_with_refine` — an OPTIONAL hook that COMPOSES the
     existing optimiser (``tools.clay_optimize.tssbo_adapter.TSsBoAdapter``,
     opt-in via ``CLAY_USE_TSSBO``) and/or the existing distance-to-sat loss
     (``clay.distance_to_sat``) to nudge the analytic seed toward measured
     KPI targets. The user asked specifically why presim used neither
     adjoint nor tssbo — this is where they belong. We do NOT reinvent
     them; the analytic seed is the warm start and the optimiser refines.
     Default ``use_optimizer=False`` so the analytic path never depends on
     the optimiser or ngspice.

The analytic model (first-order square-law MOS)
-----------------------------------------------

For a saturated MOS in strong inversion::

    Id = 0.5 * KP * (W / L) * Vov^2          (Vov = Vgs - Vth)
    gm = KP * (W / L) * Vov = 2 * Id / Vov
    gm / Id = 2 / Vov                          (operating-point invariant)

To carry the SAME device (same gm/Id, hence same Vov_design intent, same
Id) from a source process (KP_src, Lmin_src, Vdd_src) onto a target process
(KP_dst, Lmin_dst, Vdd_dst):

  * **Channel length L.** Raise to the target ``Lmin`` floor, but preserve
    the *deliberate* long-channel ratio the designer chose for ro/gain.
    A device drawn at L = k * Lmin_src (k = the design's L/Lmin ratio) maps
    to L_dst = k * Lmin_dst, then clamped up to Lmin_dst.

  * **Overdrive Vov.** The operating point fixes Vov_src = 2 / (gm/Id). On
    the target process we scale the *available* overdrive by the headroom
    ratio of the two supplies minus their thresholds (a higher supply with
    a higher Vth gives a different overdrive budget). The headroom-scaled
    overdrive is::

        Vov_dst = Vov_src * (Vdd_dst - Vth_dst) / (Vdd_src - Vth_src)

    (per-type Vth; nmos uses vth_n, pmos uses vth_p.)

  * **Width W.** Solve Id = 0.5*KP*(W/L)*Vov^2 for W on both processes and
    require Id_dst = Id_src (preserve the branch current the gm/Id budget
    set). Eliminating Id::

        W_dst = W_src * (KP_src / KP_dst)
                      * (L_dst / L_src)
                      * (Vov_src / Vov_dst)^2

    This is the load-bearing formula. Read it as: a weaker process (smaller
    KP) needs proportionally more width; a longer channel needs more width
    to hold the same current; a smaller overdrive needs more width (W ~ 1/Vov^2
    at fixed Id).

  * **Multiplicity m.** Carried through unchanged (it encodes a layout/area
    split decision, not a process quantity). The *effective* width m*W
    scales by the W formula; we apply the scale to the per-finger W and keep
    m fixed, so the effective width tracks.

  * **Bias / overdrive voltages.** A ``.param`` that names a voltage (a bias
    node target, an explicit Vov) is rescaled by the supply ratio with a
    threshold offset::

        V_dst = Vth_dst + (V_src - Vth_src) * (Vdd_dst / Vdd_src)

    i.e. the part of the voltage above threshold scales with the supply; the
    threshold floor is replaced by the target threshold. This is first-order
    and is exactly the class of value the optional refine hook exists to
    correct against a real simulator.

Honesty about what retargets and what does not
----------------------------------------------

W and L retarget analytically and land close (the square-law W formula is
the same physics the hand derivation used). What will NOT match the
committed values exactly: (a) absolute bias voltages, which depend on the
full bias-tree DC solution, not a one-device square-law; (b) layout-driven
multiplicity splits (e.g. fold W=20u carried as W=5u/m=8) — these are area
decisions the analytic pass deliberately leaves to the layout/optimiser
stage; (c) margin pads the designer hand-tuned on a real sweep. Those are
the params the refine hook is for. We do not tune constants to fake a fit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional


# ─── Engineering-suffix number parsing ────────────────────────────────


_SI_SUFFIX: dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "x": 1e6,  # spice alias for meg
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}

# A number with an optional engineering suffix. Longest suffix ("meg")
# must be tried before the single-char ones, so we match the alpha tail
# greedily and resolve it against the table below.
_NUM_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*$"
)


def parse_eng(text: str) -> float:
    """Parse a SPICE engineering-notation number to a float in SI units.

    Accepts ``"4.4u"`` -> ``4.4e-6``, ``"1n"`` -> ``1e-9``, ``"2"`` -> ``2.0``,
    ``"1.5meg"`` -> ``1.5e6``, ``"3.3"`` -> ``3.3``. A trailing unit-ish
    suffix beyond the SI letter (e.g. ``"2uF"`` -> the ``"F"`` is ignored as
    a unit tag) is tolerated: only the leading SI multiplier letter(s) are
    consumed. Raises ``ValueError`` on a non-numeric token.
    """
    if text is None:
        raise ValueError("cannot parse None as a number")
    s = str(text).strip()
    m = _NUM_RE.match(s)
    if not m:
        raise ValueError(f"not an engineering number: {text!r}")
    mantissa = float(m.group(1))
    suffix = m.group(2).lower()
    if not suffix:
        return mantissa
    # Try the multi-char "meg" first, then a single leading letter. Any
    # remaining characters are treated as a (ignored) unit tag.
    if suffix.startswith("meg"):
        return mantissa * _SI_SUFFIX["meg"]
    head = suffix[0]
    mult = _SI_SUFFIX.get(head)
    if mult is None:
        # No recognised SI letter; treat the whole suffix as a unit tag.
        return mantissa
    return mantissa * mult


def format_eng(value: float, *, ref: Optional[str] = None) -> str:
    """Render an SI float back to a SPICE micron-suffix string.

    The retarget surface deals in geometry (W/L, microns) and small
    voltages. To keep the emitted ``.param`` readable and diff-friendly
    against the committed file we render lengths/widths with the ``u``
    (micro) suffix when the magnitude is in the micron-to-millimetre band,
    fall back to plain decimal for O(1) values (multiplicities, voltages),
    and use ``n``/``p`` for sub-micron magnitudes.

    ``ref`` is the original source string; when it carried a suffix we
    prefer to keep the same suffix family so a ``"4.4u"`` source stays a
    ``u`` value rather than flipping to ``"4400n"``.
    """
    if value == 0.0:
        return "0"
    av = abs(value)
    # Honour the source suffix family when supplied.
    suffix_letter = None
    if ref:
        rm = _NUM_RE.match(str(ref).strip())
        if rm and rm.group(2):
            head = rm.group(2).lower()
            if head.startswith("meg"):
                suffix_letter = "meg"
            elif head[0] in _SI_SUFFIX:
                suffix_letter = head[0]
    if suffix_letter:
        mult = _SI_SUFFIX["meg"] if suffix_letter == "meg" else _SI_SUFFIX[suffix_letter]
        scaled = value / mult
        return f"{_trim(scaled)}{suffix_letter}"
    # No source suffix: pick by magnitude.
    if av >= 1e-3:
        # >= 1 mm in length terms, or a plain O(1) number (voltage / m).
        if av >= 1.0:
            return _trim(value)
        return f"{_trim(value / 1e-6)}u"
    if av >= 1e-6:
        return f"{_trim(value / 1e-6)}u"
    if av >= 1e-9:
        return f"{_trim(value / 1e-9)}n"
    return f"{_trim(value / 1e-12)}p"


def _trim(x: float) -> str:
    """Compact float rendering: drop trailing zeros, keep up to 6 sig-figs."""
    s = f"{x:.6g}"
    return s


# ─── PDK parameter model ──────────────────────────────────────────────


@dataclass(frozen=True)
class PdkParams:
    """First-order process constants for the analytic retarget.

    Fields:
      * ``vdd``   — nominal supply (V).
      * ``vth_n`` — NMOS threshold (V).
      * ``vth_p`` — PMOS threshold magnitude (V, positive).
      * ``kp_n``  — NMOS transconductance parameter mu_n*Cox (uA/V^2).
      * ``kp_p``  — PMOS transconductance parameter mu_p*Cox (uA/V^2).
      * ``lmin``  — minimum channel length (um).

    KP is stored in uA/V^2 (matching the ``fc_ota.cir`` header's
    ``unCox=270uA/V2`` / ``upCox=70uA/V2`` convention) so the presets read
    straight off the design notes. The retarget only uses KP ratios, so the
    unit cancels — but keeping uA/V^2 makes the presets auditable against
    the source comment.
    """

    vdd: float
    vth_n: float
    vth_p: float
    kp_n: float  # uA/V^2
    kp_p: float  # uA/V^2
    lmin: float  # um

    def kp_for(self, kind: str) -> float:
        """KP for an ``nmos``/``pmos`` (or ``n``/``p``) device kind."""
        k = kind.lower()
        if k.startswith("p"):
            return self.kp_p
        return self.kp_n

    def vth_for(self, kind: str) -> float:
        """Threshold magnitude for an ``nmos``/``pmos`` device kind."""
        k = kind.lower()
        if k.startswith("p"):
            return self.vth_p
        return self.vth_n


def VPDK180() -> PdkParams:
    """The fake ``vpdk180nm`` 1.8 V process the cirona designs were drawn on.

    Source: ``circuits/fc_ota/fc_ota.cir`` header — ``VDD = 1.8 V``,
    ``Lmin = 180 nm``, Step-3 sizing notes ``unCox = 270 uA/V2`` /
    ``upCox = 70 uA/V2``, Step-4 ``Vth_n ~ 0.35 V``. The PMOS threshold is
    not stated explicitly; ``|Vth_p| ~ 0.45 V`` is the standard 180 nm pch
    value and is documented here as an assumption (it only enters the
    bias-voltage rescale, which the refine hook corrects against sim).
    """
    return PdkParams(
        vdd=1.8,
        vth_n=0.35,   # fc_ota.cir Step-4 ("PDK Vth_n~0.35V confirmed by sub-TB")
        vth_p=0.45,   # ASSUMPTION: standard 180nm |Vth_p|; not in the source header
        kp_n=270.0,   # fc_ota.cir Step-3 unCox=270uA/V2
        kp_p=70.0,    # fc_ota.cir Step-3 upCox=70uA/V2
        lmin=0.18,
    )


def GF180_3P3() -> PdkParams:
    """GlobalFoundries gf180mcu 3.3 V process (nmos_3p3 / pmos_3p3).

    Source / assumptions (first-order; documented honestly):
      * ``vdd = 3.3``, ``lmin = 0.28`` um — the gf180mcu 3.3 V device
        minimum (the repo gf180 deck convention; ``circuits/fc_ota/
        DESIGN.md`` draws the input pair at L = 0.6 um = ~2.1*Lmin).
      * ``vth_n = 0.69``, ``vth_p = 0.78`` — MEASURED on real ngspice at
        Id = 10 uA, quoted verbatim in ``DESIGN.md`` ("Vth_n ~= 0.69 V,
        |Vth_p| ~= 0.78 V").
      * ``kp_n = 130``, ``kp_p = 35`` uA/V^2 — first-order mu*Cox for the
        gf180mcu 3.3 V devices. ASSUMPTION: the gf180mcu 3.3 V oxide is
        thicker than the 180 nm core oxide, so mu*Cox is roughly half the
        vpdk180 values; 130 / 35 is the order-of-magnitude estimate used as
        the analytic seed. The ratio KP_src/KP_dst (270/130 ~ 2.08,
        70/35 = 2.0) is what drives the width rescale; the absolute value is
        only a seed and is exactly what the refine hook tunes against the
        real device.
    """
    return PdkParams(
        vdd=3.3,
        vth_n=0.69,   # DESIGN.md measured
        vth_p=0.78,   # DESIGN.md measured
        kp_n=130.0,   # ASSUMPTION: first-order gf180 3.3V mu_n*Cox seed
        kp_p=35.0,    # ASSUMPTION: first-order gf180 3.3V mu_p*Cox seed
        lmin=0.28,
    )


# ─── Param-key classification ─────────────────────────────────────────


def _classify_key(key: str) -> tuple[str, str]:
    """Return ``(category, kind)`` for a ``.param`` key.

    category ∈ {"W", "L", "m", "V", "other"}; kind ∈ {"nmos","pmos","?"}.

    Classification is structural on the key name (the ``W_*`` / ``L_*`` /
    ``m_*`` convention the cirona ``.cir`` files use), plus a name-substring
    heuristic for the device kind so the right KP/Vth is selected:

      * a key containing ``p`` markers (``pcasc``, ``fold``, ``pad_p``,
        ``pmirror``, ``vbcp``) -> pmos;
      * ``n``/diff/mirror/bias/tail -> nmos;
      * fall back to nmos when ambiguous (the dominant device family).
    """
    k = key.lower()
    # Category by prefix.
    if k.startswith("w_") or k == "w":
        cat = "W"
    elif k.startswith("l_") or k == "l":
        cat = "L"
    elif k.startswith("m_") or k == "m":
        cat = "m"
    elif k.startswith("v") or "vbc" in k or "vbias" in k or "vov" in k:
        cat = "V"
    else:
        cat = "other"

    # Device kind by name substring.
    pmos_markers = ("pcasc", "fold", "pad_p", "_p", "pmir", "vbcp", "pch", "pmos", "_pmos")
    nmos_markers = ("ncasc", "nmir", "diff", "bias", "tail", "pad_n", "vbcn", "nch", "nmos")
    kind = "?"
    if any(mark in k for mark in pmos_markers):
        kind = "pmos"
    if any(mark in k for mark in nmos_markers):
        # nmos markers win only if no strong pmos marker already matched a
        # pmos-specific token; "fold"/"pcasc" are unambiguously pmos in
        # these circuits, so keep pmos when both somehow match.
        if kind != "pmos":
            kind = "nmos"
    if kind == "?":
        kind = "nmos"
    return cat, kind


# ─── The analytic retarget ────────────────────────────────────────────


@dataclass(frozen=True)
class RetargetReport:
    """Side-channel diagnostics from a retarget (not returned by the
    primary ``retarget_params`` surface, available via
    :func:`retarget_params_with_report`)."""

    retargeted: dict[str, str]
    width_scale: dict[str, float]   # per-W-key applied scale factor
    length_floor_hits: tuple[str, ...]  # L keys raised to dst.lmin


def _retarget_core(
    params: Mapping[str, str],
    src: PdkParams,
    dst: PdkParams,
) -> RetargetReport:
    """Shared engine for :func:`retarget_params` /
    :func:`retarget_params_with_report`."""
    # First pass: parse every key, capture L source values so a W key can
    # consult the (possibly floored) target L of its own device.
    parsed: dict[str, float] = {}
    for key, raw in params.items():
        try:
            parsed[key] = parse_eng(raw)
        except ValueError:
            parsed[key] = math.nan  # symbolic / non-numeric — pass through

    # Pre-compute the per-key target L (in um). A W key reuses the L of the
    # device it shares a suffix with (W_diff <-> L_diff), so we resolve
    # target L per *device suffix*, not just globally.
    def _device_suffix(key: str) -> str:
        k = key.lower()
        for pre in ("w_", "l_", "m_"):
            if k.startswith(pre):
                return k[len(pre):]
        return k

    # Map device-suffix -> (src_L_um, dst_L_um, floored?).
    l_by_suffix: dict[str, tuple[float, float, bool]] = {}
    for key, val in parsed.items():
        cat, _kind = _classify_key(key)
        if cat != "L" or math.isnan(val):
            continue
        suffix = _device_suffix(key)
        l_src_um = val / 1e-6  # parse_eng gave SI metres; back to um
        # Preserve the deliberate L/Lmin ratio, then floor at dst.lmin.
        ratio = l_src_um / src.lmin if src.lmin > 0 else 1.0
        l_dst_um = ratio * dst.lmin
        floored = False
        if l_dst_um < dst.lmin:
            l_dst_um = dst.lmin
            floored = True
        l_by_suffix[suffix] = (l_src_um, l_dst_um, floored)

    out: dict[str, str] = {}
    width_scale: dict[str, float] = {}
    floor_hits: list[str] = []

    for key, raw in params.items():
        val = parsed[key]
        cat, kind = _classify_key(key)

        if math.isnan(val):
            out[key] = raw  # symbolic: preserve verbatim
            continue

        if cat == "m":
            # Multiplicity is a layout split decision, not a process
            # quantity — carry through unchanged.
            out[key] = raw
            continue

        if cat == "L":
            suffix = _device_suffix(key)
            _l_src, l_dst_um, floored = l_by_suffix[suffix]
            if floored:
                floor_hits.append(key)
            out[key] = format_eng(l_dst_um * 1e-6, ref=raw)
            continue

        if cat == "W":
            suffix = _device_suffix(key)
            # Target / source L for this device (falls back to Lmin-floored
            # min if the device had no explicit L param).
            if suffix in l_by_suffix:
                l_src_um, l_dst_um, _fl = l_by_suffix[suffix]
            else:
                l_src_um, l_dst_um = src.lmin, dst.lmin
            kp_src = src.kp_for(kind)
            kp_dst = dst.kp_for(kind)
            vth_src = src.vth_for(kind)
            vth_dst = dst.vth_for(kind)
            # Operating-point overdrive intent: we do not know each device's
            # exact gm/Id here, but the headroom ratio is the same for every
            # device on a rail, so the Vov scale factor is shared:
            #   Vov_dst / Vov_src = (Vdd_dst - Vth_dst) / (Vdd_src - Vth_src)
            head_src = src.vdd - vth_src
            head_dst = dst.vdd - vth_dst
            vov_ratio = (head_dst / head_src) if head_src > 0 else 1.0
            #   W_dst = W_src * (KP_src/KP_dst) * (L_dst/L_src) * (Vov_src/Vov_dst)^2
            l_ratio = (l_dst_um / l_src_um) if l_src_um > 0 else 1.0
            kp_ratio = (kp_src / kp_dst) if kp_dst > 0 else 1.0
            inv_vov2 = (1.0 / vov_ratio) ** 2 if vov_ratio > 0 else 1.0
            scale = kp_ratio * l_ratio * inv_vov2
            width_scale[key] = scale
            w_src_si = val  # SI metres
            out[key] = format_eng(w_src_si * scale, ref=raw)
            continue

        if cat == "V":
            # Bias / overdrive voltage rescale:
            #   V_dst = Vth_dst + (V_src - Vth_src) * (Vdd_dst / Vdd_src)
            vth_src = src.vth_for(kind)
            vth_dst = dst.vth_for(kind)
            supply_ratio = dst.vdd / src.vdd if src.vdd > 0 else 1.0
            v_dst = vth_dst + (val - vth_src) * supply_ratio
            out[key] = format_eng(v_dst, ref=raw)
            continue

        # Unknown category: pass through unchanged.
        out[key] = raw

    return RetargetReport(
        retargeted=out,
        width_scale=width_scale,
        length_floor_hits=tuple(floor_hits),
    )


def retarget_params(
    params: Mapping[str, str],
    src: PdkParams,
    dst: PdkParams,
) -> dict[str, str]:
    """Retarget a source ``.param`` map from ``src`` PDK onto ``dst`` PDK.

    Default analytic path (no simulator, no optimiser). Holds each device's
    operating point (gm/Id, hence Vov intent) and branch current while
    swapping the process constants:

      * W rescaled by ``(KP_src/KP_dst) * (L_dst/L_src) * (Vov_src/Vov_dst)^2``
      * L raised to ``dst.lmin`` (preserving the design's L/Lmin ratio)
      * m carried through (a layout split, not a process quantity)
      * bias voltages rescaled by the supply ratio with a Vth offset

    Returns a NEW dict with the same keys and retargeted string values.
    Symbolic / non-numeric values pass through verbatim so the design's
    symbolic structure (``W=W_diff`` references) is preserved.
    """
    return _retarget_core(params, src, dst).retargeted


def retarget_params_with_report(
    params: Mapping[str, str],
    src: PdkParams,
    dst: PdkParams,
) -> RetargetReport:
    """As :func:`retarget_params` but also returns the per-key width-scale
    factors and the list of L keys that were floored to ``dst.lmin``."""
    return _retarget_core(params, src, dst)


# ─── Optional optimiser-refine hook (composes existing machinery) ─────


def _build_bounds(seed_params: Mapping[str, str], rel_window: float):
    """Build a ``SizingBounds`` list around the analytic seed for the
    numeric (W/L/V) knobs. Lazy-imports the harness type so the analytic
    path never pulls the optimiser deps in.

    Returns ``(bounds, knob_keys)`` or ``(None, [])`` when the harness
    import fails (optimiser machinery absent).
    """
    try:
        from tools.clay_optimize.harness import SizingBounds
    except Exception:  # noqa: BLE001 — optimiser machinery optional
        return None, []
    bounds = []
    knobs: list[str] = []
    for key, raw in seed_params.items():
        cat, _kind = _classify_key(key)
        if cat not in ("W", "L", "V"):
            continue
        try:
            v = parse_eng(raw)
        except ValueError:
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        lo = v * (1.0 - rel_window)
        hi = v * (1.0 + rel_window)
        bounds.append(SizingBounds(knob=key, low=lo, high=hi, log_scale=True))
        knobs.append(key)
    return bounds, knobs


def retarget_with_refine(
    params: Mapping[str, str],
    src: PdkParams,
    dst: PdkParams,
    *,
    evaluate: Optional[Callable[[Mapping[str, str]], Mapping[str, float]]] = None,
    use_optimizer: bool = False,
    spec_targets: Optional[Mapping[str, tuple[str, float]]] = None,
    max_iters: int = 20,
    rel_window: float = 0.5,
    seed: int = 0,
) -> dict[str, str]:
    """Retarget, then OPTIONALLY refine the analytic seed against KPI targets.

    The analytic seed always comes first via :func:`retarget_params`. When
    ``use_optimizer`` is False (the default) this is a pure no-op over the
    analytic result — tests and the common path need neither the optimiser
    nor ngspice.

    When ``use_optimizer`` AND an ``evaluate`` callback are both supplied,
    the function COMPOSES the existing machinery (it does NOT reinvent BO or
    the loss):

      * ``tools.clay_optimize.tssbo_adapter.TSsBoAdapter`` — the real tSS-BO
        optimiser behind its 4-layer opt-in (env ``CLAY_USE_TSSBO`` + the
        ``enabled_by_cli`` flag + the package import + a budget). The
        analytic retarget is the warm-start seed; bounds are a ±``rel_window``
        box around the seed (log-scale).
      * ``clay.distance_to_sat`` — turns the ``evaluate`` KPI dict into a
        continuous satisfaction loss against ``spec_targets`` so the
        optimiser sees a dense reward instead of a 0/1 pass.

    Every import is gated: if the optimiser is absent, mis-opted-in, or the
    distance loss import fails, the function returns the analytic seed
    unchanged (graceful degradation — a missing optimiser never breaks the
    analytic path). ``evaluate`` maps a candidate ``.param`` dict to a
    ``{metric: value}`` KPI dict (typically a thin wrapper over the circuit's
    ``presim.py`` / an ngspice run); ``spec_targets`` maps a metric name to
    an ``(op, target)`` pair (e.g. ``{"dc_gain_db": (">=", 80.0)}``).

    Returns a ``.param`` dict (same keys as ``params``).
    """
    seed_params = retarget_params(params, src, dst)
    if not use_optimizer or evaluate is None:
        return seed_params

    # ── Compose the distance-to-sat loss (existing module) ────────────
    try:
        from clay.distance_to_sat import compute_spec_distance
    except Exception:  # noqa: BLE001
        compute_spec_distance = None  # type: ignore[assignment]

    def _loss(kpis: Mapping[str, float]) -> float:
        if not spec_targets:
            return 0.0
        if compute_spec_distance is None:
            # Fall back to a plain hinge sum if the loss module is absent.
            total = 0.0
            for name, (op, tgt) in spec_targets.items():
                m = kpis.get(name)
                if m is None:
                    total += 1e9
                    continue
                if op in (">=", ">"):
                    total += max(0.0, tgt - m)
                elif op in ("<=", "<"):
                    total += max(0.0, m - tgt)
                else:
                    total += abs(m - tgt)
            return total
        total = 0.0
        for name, (op, tgt) in spec_targets.items():
            m = kpis.get(name)
            d = compute_spec_distance(
                m if m is not None else None, float(tgt), op,
            )
            total += d if math.isfinite(d) else 1e9
        return total

    # ── Compose the existing tSS-BO optimiser (gated) ─────────────────
    bounds, knobs = _build_bounds(seed_params, rel_window)
    if not bounds:
        # Optimiser harness types unavailable — return the analytic seed.
        return seed_params

    try:
        from tools.clay_optimize.tssbo_adapter import build_tssbo_adapter
        from tools.clay_optimize.harness import OptimizerHistory
    except Exception:  # noqa: BLE001
        return seed_params

    try:
        optimizer = build_tssbo_adapter(bounds, seed=seed)
    except Exception:  # noqa: BLE001
        # The 4-layer opt-in refused (env not set, package missing, budget).
        # Graceful degradation: return the analytic seed unchanged.
        return seed_params

    # Run the refine loop. The optimiser proposes a numeric knob dict; we
    # overlay it onto the seed (keeping the symbolic / m params) and call
    # the user's evaluate -> KPI dict -> distance loss -> reward.
    try:
        history = OptimizerHistory()
    except Exception:  # noqa: BLE001
        return seed_params

    best_params = dict(seed_params)
    best_loss = math.inf
    # Evaluate the seed first so the optimiser has a baseline.
    try:
        seed_kpis = evaluate(seed_params)
        best_loss = _loss(seed_kpis)
    except Exception:  # noqa: BLE001
        best_loss = math.inf

    for _ in range(max(1, max_iters)):
        try:
            proposal = optimizer.propose(history)
        except Exception:  # noqa: BLE001 — budget tripped or backend error
            break
        candidate = dict(seed_params)
        for knob in knobs:
            if knob in proposal:
                candidate[knob] = format_eng(
                    float(proposal[knob]), ref=seed_params.get(knob),
                )
        try:
            kpis = evaluate(candidate)
        except Exception:  # noqa: BLE001
            kpis = {}
        loss = _loss(kpis)
        reward = -loss if math.isfinite(loss) else -1e9
        feasible = math.isfinite(loss) and loss <= 0.0
        if loss < best_loss:
            best_loss = loss
            best_params = candidate
        # Push the observation back into the surrogate (existing API).
        try:
            from tools.clay_optimize.harness import OptimizerHistoryEntry
            entry = OptimizerHistoryEntry(
                iteration=len(getattr(history, "entries", []) or []),
                sizing={k: float(proposal.get(k, 0.0)) for k in knobs},
                measures=dict(kpis),
                violations=(),
                reward=reward,
                is_feasible=feasible,
            )
            optimizer.update(history, entry.sizing, entry)
        except Exception:  # noqa: BLE001
            break

    return best_params


__all__ = [
    "PdkParams",
    "VPDK180",
    "GF180_3P3",
    "parse_eng",
    "format_eng",
    "retarget_params",
    "retarget_params_with_report",
    "retarget_with_refine",
    "RetargetReport",
]
