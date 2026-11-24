"""Simulation-based PDK characterizer: MEASURE Vth and KP by running ngspice on a
single-transistor testbench, to complement (and refine) the static BSIM card-parse
characterizer in ``clay.frontend.pdk_characterize``.

Why this exists
---------------
The static parser (``characterize_pdk``) reads the *leading numeric* of each BSIM
parameter expression off the model card. That nominal is sometimes wrong or
missing -- e.g. open_pdk parses nmos KP=0, open_pdk pmos KP comes out odd -- because
the real device transconductance is set by the full BSIM solution (binning, mobility
degradation, body effect, short-channel), not by ``u0``/``toxe`` alone. Where the PDK
is *ngspice-runnable* we can do better: bias a real device and measure.

What it does NOT do
-------------------
It does not fabricate numbers. If the simulator or PDK is not available, or the PDK
ships HSPICE-only models that ngspice cannot parse (every "real" commercial target:
open_pdk/65/28, open_pdk, open_pdk), :func:`sim_characterize_pdk` raises
:class:`SimCharacterizeUnavailable` and the caller falls back to the static parse.

Mechanism
---------
For each of NMOS / PMOS:

  * Instantiate the PDK core device as a single transistor with a known W and L
    (L = a few x Lmin, W ~ 10 x L), drain held at Vdd (saturation), gate swept
    0..Vdd.  PMOS is built mirrored about Vdd so all measured quantities are taken
    as magnitudes.
  * ``Id`` is measured as the supply-branch current (``-i(Vds)``) at every sweep
    point -- no internal device handle needed, so the extraction is independent of
    how the PDK names its devices.
  * **Vth** by the max-gm linear-extrapolation method: find the Vgs of peak
    transconductance gm = dId/dVgs, draw the tangent there, and read its
    Vgs-intercept; subtract half the drain bias as the standard Vds/2 correction
    so the extrapolated value lands near the model Vth0 rather than Vth0+Vds/2.
  * **KP** (= mu*Cox, uA/V^2) from the saturation square law evaluated well into
    saturation (Vgs = Vdd):  Id = 0.5*KP*(W/L)*(Vgs-Vth)^2  =>
    KP = 2*Id*L / (W*(Vgs-Vth)^2).

The extraction math (:func:`extract_vth_linear`, :func:`extract_kp_satlaw`) is pure
and is unit-tested against a SYNTHETIC Id-Vgs curve generated from a known Vth/KP,
so the recovery accuracy is provable without a simulator.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ─── Errors ───────────────────────────────────────────────────────────


class SimCharacterizeUnavailable(RuntimeError):
    """Raised when the sim-characterize path cannot run for this PDK in this env.

    Carries a specific reason: simulator not installed, the PDK is not present /
    not ngspice-runnable, or the device model could not be exercised. The caller
    is expected to fall back to the static card-parse characterizer.
    """


# ─── PDK -> ngspice binding (THROUGH the harness, never hardcoded paths) ──

# Map a resolvable PDK id to (harness marker, core nmos device, core pmos device,
# default supply, units flag). ``scale_micron`` says whether the deck must carry
# ``.option scale=1.0u`` (so W/L are written in microns) -- this matches the
# committed deck convention per PDK (gf180 uses scale; open_pdk passes meters).
_SIM_PDKS: Dict[str, dict] = {
    "gf180mcu": {
        "marker": "* @GF180_PDK corner=tt@",
        "nmos": "nmos_3p3",
        "pmos": "pmos_3p3",
        "vdd": 3.3,
        "lmin": 0.28,
        "scale_micron": True,
        # gf180 core devices are .subckt d g s b with w/l params.
        "instance": "subckt",
    },
    "open_pdk": {
        "marker": "* @external_pdk corner=tt@",
        "nmos": "nmos_3p3",
        "pmos": "pmos_3p3",
        "vdd": 1.8,
        "lmin": 0.18,
        "scale_micron": False,
        "instance": "subckt",
    },
}

# PDK ids whose name maps onto a sim-capable adapter even when passed via a
# directory whose basename is not exactly the harness id.
_DIRNAME_HINTS: Tuple[Tuple[str, str], ...] = (
    ("gf180", "gf180mcu"),
    ("open_pdk", "open_pdk"),
    ("external_model", "open_pdk"),
)


def _detect_pdk_id(pdk_dir: Path, pdk_marker: Optional[str]) -> Optional[str]:
    """Best-effort map a pdk_dir / explicit marker to a sim-capable PDK id."""
    if pdk_marker:
        m = pdk_marker.lower()
        if "gf180" in m:
            return "gf180mcu"
        if "open_pdk" in m:
            return "open_pdk"
    name = pdk_dir.name.lower()
    for needle, pid in _DIRNAME_HINTS:
        if needle in name:
            return pid
    # also look one level up (callers may pass .../models/ngspice)
    for part in pdk_dir.parts:
        pl = part.lower()
        for needle, pid in _DIRNAME_HINTS:
            if needle in pl:
                return pid
    return None


def _harness():
    """Import the marker-based PDK harness lazily (keeps the static path light)."""
    from circuits import _harness as H  # noqa: WPS433 — intentional local import
    return H


def _sim_ready(pdk_id: str) -> bool:
    H = _harness()
    try:
        return bool(H.sim_ready(pdk_id))
    except Exception:  # noqa: BLE001 — harness raises PdkNotFound subtypes
        return False


# ─── Pure extraction math (unit-tested on synthetic curves) ───────────


def _gradient(x: Sequence[float], y: Sequence[float]) -> List[float]:
    """Central-difference derivative dy/dx on a (monotone-x) sampled curve."""
    n = len(x)
    g = [0.0] * n
    for i in range(n):
        if i == 0:
            g[i] = (y[1] - y[0]) / (x[1] - x[0])
        elif i == n - 1:
            g[i] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
        else:
            g[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return g


def _linfit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Ordinary least-squares fit ``y = a*x + b``; returns ``(a, b)``."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("degenerate x in linear fit")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def extract_vth_linear(
    vgs: Sequence[float],
    idd: Sequence[float],
    *,
    vds: float = 0.0,
) -> float:
    """Threshold by sqrt(Id) linear extrapolation on a SATURATION Id-Vgs sweep.

    In saturation the square law gives ``sqrt(Id) = sqrt(0.5*KP*W/L) * (Vgs - Vth)``,
    i.e. ``sqrt(Id)`` is a straight line in ``Vgs`` whose Vgs-axis intercept is Vth
    EXACTLY (no Vds/2 overshoot, unlike the linear-region max-gm tangent on Id
    itself). We locate the strong-inversion segment via the peak slope of
    ``sqrt(Id)`` (equivalently peak gm/sqrt(Id)), fit a line to the points around
    and above it, and return the intercept. This recovers Vth on a pure square law
    to grid precision and is robust on real curves (the fit window sits in strong
    inversion, away from the subthreshold tail and from velocity-saturation roll-off
    at the very top).

    ``vds`` is accepted for API symmetry; for a saturation sqrt(Id) fit no Vds/2
    correction is applied (the intercept is already the saturation Vth).
    """
    if len(vgs) != len(idd) or len(vgs) < 3:
        raise ValueError("need >=3 aligned (vgs, idd) points")
    sq = [math.sqrt(i) if i > 0 else 0.0 for i in idd]
    # slope of sqrt(Id) per step; strong inversion is where this is largest.
    slope = _gradient(vgs, sq)
    k = max(range(1, len(slope)), key=lambda i: slope[i])
    # Fit the strong-inversion segment: points from just before the peak-slope
    # index up to (but not including) the very last point if a roll-off is present.
    lo = max(1, k - 2)
    hi = len(vgs)
    xs = list(vgs[lo:hi])
    ys = list(sq[lo:hi])
    if len(xs) < 2:
        xs, ys = list(vgs), list(sq)
    a, b = _linfit(xs, ys)
    if a <= 0:
        raise ValueError("non-positive sqrt(Id) slope; device never turned on")
    vth = -b / a
    _ = vds  # documented no-op for saturation sweeps
    return float(vth)


def extract_vth_constant_current(
    vgs: Sequence[float],
    idd: Sequence[float],
    *,
    w_um: float,
    l_um: float,
    iref_per_square: float = 1e-7,
) -> float:
    """Threshold by the constant-current method: the Vgs at which the drain
    current crosses a geometry-normalised reference ``Id_th = iref_per_square *
    (W/L)``.

    ``iref_per_square = 0.1 uA`` (1e-7 A) per W/L square is the standard textbook
    constant-current threshold definition. This is the method the gf180
    ``DESIGN.md`` quotes its measured Vth at, and it is far more robust on a REAL
    BSIM curve than tangent extrapolation -- it reads a single well-defined point
    on the turn-on knee rather than fitting a slope that mobility degradation
    bends at high Vgs. The crossing Vgs is linearly interpolated between samples.
    """
    if len(vgs) != len(idd) or len(vgs) < 3:
        raise ValueError("need >=3 aligned (vgs, idd) points")
    if w_um <= 0 or l_um <= 0:
        raise ValueError("W and L must be positive")
    id_th = iref_per_square * (w_um / l_um)
    for j in range(1, len(vgs)):
        if idd[j] >= id_th:
            x0, x1 = vgs[j - 1], vgs[j]
            y0, y1 = idd[j - 1], idd[j]
            if y1 == y0:
                return float(x1)
            return float(x0 + (id_th - y0) * (x1 - x0) / (y1 - y0))
    raise ValueError(
        f"current never reached Id_th={id_th:.3g} A; raise Vdd or lower iref"
    )


def extract_kp_satlaw(
    vgs_eval: float,
    id_eval: float,
    vth: float,
    *,
    w_um: float,
    l_um: float,
) -> float:
    """mu*Cox (uA/V^2) from the saturation square law at one well-saturated point.

        Id = 0.5 * KP * (W/L) * (Vgs - Vth)^2
        => KP = 2 * Id * L / (W * (Vgs - Vth)^2)

    ``id_eval`` in amps, geometry in microns (the W/L *ratio* is unitless so the
    micron units cancel); the 1e6 converts A/V^2 to uA/V^2. Caller passes
    magnitudes for PMOS.
    """
    vov = vgs_eval - vth
    if vov <= 0:
        raise ValueError(f"Vov={vov:.4g} <= 0: evaluation point not in strong inversion")
    if w_um <= 0 or l_um <= 0:
        raise ValueError("W and L must be positive")
    kp = 2.0 * id_eval * (l_um / w_um) / (vov * vov)   # A/V^2
    return float(kp * 1e6)                              # uA/V^2


def _measure_from_curve(
    vgs: Sequence[float],
    idd: Sequence[float],
    *,
    vdd: float,
    w_um: float,
    l_um: float,
) -> Tuple[float, float]:
    """Run both extractions on one Id-Vgs sweep. Returns ``(vth, kp_uA_V2)``.

    Vth uses the constant-current method (robust on real BSIM curves and the
    method the gf180 DESIGN.md quotes); KP is read from the saturation square law
    at the top of the sweep (deepest saturation = the cleanest square-law point).
    """
    vth = extract_vth_constant_current(vgs, idd, w_um=w_um, l_um=l_um)
    # evaluate KP at the most-saturated swept point (largest Vgs).
    j = max(range(len(vgs)), key=lambda i: vgs[i])
    kp = extract_kp_satlaw(vgs[j], idd[j], vth, w_um=w_um, l_um=l_um)
    return vth, kp


# ─── ngspice deck build + run (uses harness ngspice_bin) ──────────────


def _build_deck(
    pdk_id: str,
    kind: str,
    *,
    vdd: float,
    w_um: float,
    l_um: float,
    npts: int,
) -> str:
    """A tiny single-transistor Id-Vgs sweep deck with the harness PDK marker.

    NMOS: source/bulk at 0, drain at Vdd, gate swept 0..Vdd.
    PMOS: source/bulk at Vdd, drain at 0, gate swept Vdd..0 (so |Vgs| rises);
          mirrored so the measured magnitudes match the NMOS extraction code.
    Id is the supply-branch current; we write it (vs the swept gate source) with
    ``wrdata`` because ``print`` of a swept vector is unreliable in -b batch mode.
    """
    cfg = _SIM_PDKS[pdk_id]
    nmos, pmos = cfg["nmos"], cfg["pmos"]
    marker = cfg["marker"]
    scale_line = ".option scale=1.0u" if cfg["scale_micron"] else ""
    step = vdd / float(max(2, npts - 1))

    if kind == "nmos":
        dev = nmos
        # drain=d at Vdd, gate=g swept, source/bulk grounded.
        lines = [
            f"* sim-characterize {pdk_id} NMOS Id-Vgs (saturation)",
            scale_line,
            marker,
            f".param VDD={vdd}",
            f".param WW={w_um}",
            f".param LL={l_um}",
            "Vds d 0 DC {VDD}",
            "Vgs g 0 DC 0",
            f"Xm d g 0 0 {dev} W={{WW}} L={{LL}}",
            ".control",
            f"dc Vgs 0 {vdd} {step}",
            "let idd = -i(Vds)",          # current sourced into the drain
            "wrdata _simchar_out.txt idd",
            "quit",
            ".endc",
            ".end",
        ]
    elif kind == "pmos":
        dev = pmos
        # source/bulk at Vdd, drain at 0; sweep gate Vdd->0 so |Vsg| rises with
        # the sweep index. We sweep the gate node from Vdd down to 0 and write
        # the magnitude of the drain current.
        lines = [
            f"* sim-characterize {pdk_id} PMOS |Id|-|Vsg| (saturation)",
            scale_line,
            marker,
            f".param VDD={vdd}",
            f".param WW={w_um}",
            f".param LL={l_um}",
            "Vs s 0 DC {VDD}",
            "Vg g 0 DC {VDD}",
            f"Xm 0 g s s {dev} W={{WW}} L={{LL}}",   # drain=0, gate=g, source=bulk=s(=Vdd)
            ".control",
            # sweep the gate from VDD down to 0; |Vsg| = VDD - Vg rises as Vg falls.
            f"dc Vg {vdd} 0 -{step}",
            "let idd = abs(i(Vs))",        # |source current| = |Id|
            "wrdata _simchar_out.txt idd",
            "quit",
            ".endc",
            ".end",
        ]
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"kind must be nmos/pmos, got {kind!r}")

    return "\n".join(ln for ln in lines if ln != "") + "\n"


def _run_sweep(
    pdk_id: str,
    kind: str,
    *,
    vdd: float,
    w_um: float,
    l_um: float,
    npts: int,
    timeout: int,
) -> Tuple[List[float], List[float], str]:
    """Build, run, and parse one Id-Vgs (or |Id|-|Vsg|) sweep.

    Returns ``(vgs_or_vsg_magnitude, id_magnitude, ngspice_cmd_str)``. Uses the
    harness ``ngspice_bin()`` (so $CLAY_NGSPICE / ngspice_con discovery is shared)
    and runs in a temp dir to avoid touching committed files.
    """
    H = _harness()
    try:
        ngspice = H.ngspice_bin()
    except Exception as exc:  # noqa: BLE001
        raise SimCharacterizeUnavailable(f"ngspice not found via harness: {exc}") from exc

    deck = _build_deck(pdk_id, kind, vdd=vdd, w_um=w_um, l_um=l_um, npts=npts)
    work = Path(tempfile.mkdtemp(prefix="clay_simchar_"))
    deck_path = work / f"_simchar_{kind}.sp"
    out_path = work / "_simchar_out.txt"
    # Inject the resolved PDK preamble at the marker via the harness builder.
    deck_path.write_text(deck, encoding="utf-8")
    runnable = H.build_deck(deck_path)
    deck_path.write_text(runnable, encoding="utf-8")
    cmd = [ngspice, "-b", deck_path.name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not out_path.is_file():
            raise SimCharacterizeUnavailable(
                f"{pdk_id} {kind}: ngspice produced no sweep output. "
                f"The PDK device {_SIM_PDKS[pdk_id][kind]!r} may not parse on ngspice. "
                f"stderr tail: {proc.stderr[-400:]!r}"
            )
        vgs, idd = _parse_wrdata(out_path.read_text(encoding="utf-8", errors="replace"))
        if len(vgs) < 3:
            raise SimCharacterizeUnavailable(
                f"{pdk_id} {kind}: sweep returned <3 points (device may not have "
                f"turned on). stdout tail: {proc.stdout[-400:]!r}"
            )
        # for PMOS the gate was swept high->low; convert the gate node value to a
        # rising |Vsg| = Vdd - Vg so the extraction code sees a standard curve.
        if kind == "pmos":
            vmag = [vdd - v for v in vgs]
            # re-sort ascending by |Vsg|
            order = sorted(range(len(vmag)), key=lambda i: vmag[i])
            vgs = [vmag[i] for i in order]
            idd = [idd[i] for i in order]
        return vgs, idd, " ".join(cmd)
    except subprocess.TimeoutExpired as exc:
        raise SimCharacterizeUnavailable(
            f"{pdk_id} {kind}: ngspice timed out after {timeout}s"
        ) from exc
    finally:
        for p in (deck_path, out_path):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            work.rmdir()
        except OSError:
            pass


_DATA_RE = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def _parse_wrdata(text: str) -> Tuple[List[float], List[float]]:
    """Parse ngspice ``wrdata`` 2-column output: ``<sweep>  <value>`` per row."""
    xs: List[float] = []
    ys: List[float] = []
    for ln in text.splitlines():
        m = _DATA_RE.match(ln)
        if not m:
            continue
        try:
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
        except ValueError:
            continue
    return xs, ys


# ─── Public surface ───────────────────────────────────────────────────


def sim_characterize_pdk(
    pdk_dir: str,
    *,
    pdk_marker: Optional[str] = None,
    vdd: Optional[float] = None,
    device: Optional[str] = None,
    params=None,
    npts: int = 34,
    timeout: int = 120,
) -> dict:
    """Measure Vth/KP for the NMOS and PMOS core devices of a runnable PDK.

    Parameters
    ----------
    pdk_dir:
        Path to the PDK directory (its basename / a passed ``pdk_marker`` selects
        the sim-capable adapter). Only PDKs the marker-based harness can drive on
        ngspice are supported (gf180mcu, open_pdk).
    pdk_marker:
        Optional explicit harness marker (e.g. ``* @GF180_PDK corner=tt@``) to
        force the adapter when the directory name is ambiguous.
    vdd:
        Supply for the sweep. Defaults to the adapter's nominal, or to a passed
        ``params`` (PdkParams) ``.vdd`` when given.
    device:
        Currently informational (the adapter fixes the core device); accepted so
        callers can record their intent.
    params:
        Optional ``PdkParams`` (e.g. from the static characterizer) used to seed
        the default Vdd; never used to fabricate the measured outputs.

    Returns
    -------
    dict with ``vth_n``, ``vth_p``, ``kp_n``, ``kp_p`` (KP in uA/V^2), ``mode``
    = "sim", and a ``provenance`` dict (measured points, W, L, the ngspice
    command, and the resolved PDK id).

    Raises
    ------
    SimCharacterizeUnavailable
        when the simulator is missing, the PDK is absent, or it is not
        ngspice-runnable (HSPICE-only commercial models). Never returns fabricated
        numbers; the caller falls back to the static parse.
    """
    d = Path(pdk_dir)
    pdk_id = _detect_pdk_id(d, pdk_marker)
    if pdk_id is None or pdk_id not in _SIM_PDKS:
        raise SimCharacterizeUnavailable(
            f"PDK at {pdk_dir!r} is not a known ngspice-runnable PDK "
            f"(supported sim PDKs: {sorted(_SIM_PDKS)}). Real commercial PDKs "
            "(open_pdk/65/28, open_pdk, open_pdk) ship HSPICE-only models that "
            "ngspice cannot parse; use the static card-parse characterizer instead."
        )
    if not _sim_ready(pdk_id):
        raise SimCharacterizeUnavailable(
            f"{pdk_id}: simulator and/or PDK not resolvable in this environment "
            "(need ngspice on PATH/$CLAY_NGSPICE and the PDK install env var). "
            "Falling back to static parse is expected."
        )

    cfg = _SIM_PDKS[pdk_id]
    if vdd is None:
        vdd = getattr(params, "vdd", None) if params is not None else None
        vdd = float(vdd) if vdd else float(cfg["vdd"])

    lmin = float(cfg["lmin"])
    l_um = max(4.0 * lmin, 1.0)      # a few x Lmin (>=1um) for a clean long-channel fit
    w_um = 10.0 * l_um               # W ~ 10x L

    vgs_n, id_n, cmd_n = _run_sweep(
        pdk_id, "nmos", vdd=vdd, w_um=w_um, l_um=l_um, npts=npts, timeout=timeout
    )
    vgs_p, id_p, cmd_p = _run_sweep(
        pdk_id, "pmos", vdd=vdd, w_um=w_um, l_um=l_um, npts=npts, timeout=timeout
    )

    vth_n, kp_n = _measure_from_curve(vgs_n, id_n, vdd=vdd, w_um=w_um, l_um=l_um)
    vth_p, kp_p = _measure_from_curve(vgs_p, id_p, vdd=vdd, w_um=w_um, l_um=l_um)

    return {
        "vth_n": vth_n,
        "vth_p": vth_p,
        "kp_n": kp_n,
        "kp_p": kp_p,
        "mode": "sim",
        "provenance": {
            "pdk_id": pdk_id,
            "vdd": vdd,
            "W_um": w_um,
            "L_um": l_um,
            "device": device,
            "nmos_device": cfg["nmos"],
            "pmos_device": cfg["pmos"],
            "sim_cmd": cmd_n,
            "sim_cmd_pmos": cmd_p,
            "nmos_curve": {"vgs": vgs_n, "id": id_n},
            "pmos_curve": {"vsg": vgs_p, "id": id_p},
            "method": "max-gm linear-extrap Vth; saturation square-law KP @ Vgs=Vdd",
        },
    }


def refine_characterization(charz, pdk_dir: str, **kw):
    """Replace a static ``PdkCharacterization``'s Vth/KP with MEASURED values.

    Takes the result of :func:`clay.frontend.pdk_characterize.characterize_pdk`
    and returns a NEW ``PdkCharacterization`` whose ``PdkParams`` carries the
    sim-measured vth_n/vth_p/kp_n/kp_p when the sim path succeeds. On any
    :class:`SimCharacterizeUnavailable` it returns the input unchanged with a note
    appended (so callers always get a usable characterization).

    ``vdd``/``lmin`` from the static params are preserved; only the four measured
    quantities are overwritten. Provenance records both the static source and the
    measured points so the override is auditable.
    """
    from clay.frontend.pdk_characterize import PdkCharacterization

    # default the sweep Vdd to the statically-parsed supply.
    kw.setdefault("params", charz.params)
    try:
        meas = sim_characterize_pdk(pdk_dir, **kw)
    except SimCharacterizeUnavailable as exc:
        note = f"sim-refine unavailable ({exc.__class__.__name__}): {exc}"
        return replace(charz, notes=tuple(charz.notes) + (note,))

    old = charz.params
    new_params = replace(
        old,
        vth_n=float(meas["vth_n"]),
        vth_p=float(meas["vth_p"]),
        kp_n=float(meas["kp_n"]),
        kp_p=float(meas["kp_p"]),
    )
    prov = dict(charz.provenance)
    prov["sim_refine"] = meas["provenance"]
    prov["static_params"] = {
        "vth_n": old.vth_n, "vth_p": old.vth_p,
        "kp_n": old.kp_n, "kp_p": old.kp_p,
    }
    note = (
        "Vth/KP overwritten by simulation-measured values "
        f"(static->sim: vth_n {old.vth_n:.3g}->{new_params.vth_n:.3g}, "
        f"kp_n {old.kp_n:.3g}->{new_params.kp_n:.3g} uA/V^2)"
    )
    return PdkCharacterization(
        pdk=charz.pdk,
        params=new_params,
        mode="sim_refined",
        provenance=prov,
        notes=tuple(charz.notes) + (note,),
    )


__all__ = [
    "SimCharacterizeUnavailable",
    "sim_characterize_pdk",
    "refine_characterization",
    "extract_vth_linear",
    "extract_vth_constant_current",
    "extract_kp_satlaw",
]
