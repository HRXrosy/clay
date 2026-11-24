"""PDK migration helper for Clay Phase 1 sizing translation.

Given a transistor sizing (W, L, finger_count) in a source PDK, suggest an
equivalent sizing for a target PDK that preserves the Pelgrom A_VT
mismatch budget:

    sigma(dVth) = A_VT / sqrt(W * L * N)

To keep sigma_target <= sigma_source we need

    (W * L * N)_target >= (A_VT_target / A_VT_source)^2 * (W * L * N)_source

We prefer to hold L fixed (only raising it to the target PDK's min_L
floor when necessary) and scale W proportionally so that the W * L * N
product satisfies the inequality.

This module is intentionally a thin, pure-Python helper that depends
only on :mod:`clay.layout.pdk_dims`. It does not modify the existing
PDK dimensional API.
"""

from dataclasses import dataclass
from typing import List

from .pdk_dims import DeviceType, Gf180mcu, PdkDims, Sky130


_PDK_REGISTRY = {
    "sky130": Sky130,
    "gf180mcu": Gf180mcu,
}


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of a PDK sizing migration request.

    Attributes
    ----------
    source_pdk, target_pdk:
        PDK name strings (matching ``PdkDims.name()``).
    source_w_um, source_l_um, source_finger_count:
        Echoed input sizing in the source PDK.
    target_w_um, target_l_um, target_finger_count:
        Suggested equivalent sizing in the target PDK.
    sigma_preserved:
        True iff the target sizing yields sigma(dVth) <= source sigma.
    notes:
        Free-form caveats (e.g. ``"min_L floor applied"``).
    """

    source_pdk: str
    target_pdk: str
    source_w_um: float
    source_l_um: float
    source_finger_count: int
    target_w_um: float
    target_l_um: float
    target_finger_count: int
    sigma_preserved: bool
    notes: List[str]


def _resolve_pdk(name: str) -> PdkDims:
    try:
        return _PDK_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown PDK name {name!r}; supported: "
            f"{sorted(_PDK_REGISTRY.keys())}"
        ) from exc


def _sigma_proxy(avt_mv_um: float, w_um: float, l_um: float, n: int) -> float:
    """Return A_VT / sqrt(W * L * N) (mV); used only for comparison."""
    area = max(w_um, 0.0) * max(l_um, 0.0) * max(n, 1)
    if area <= 0.0:
        return float("inf")
    return avt_mv_um / (area ** 0.5)


def migrate_sizing(
    source_w_um: float,
    source_l_um: float,
    source_finger_count: int,
    source_pdk_name: str,
    target_pdk_name: str,
    device_type: DeviceType = DeviceType.NMOS,
) -> MigrationResult:
    """Translate a sizing from ``source_pdk_name`` to ``target_pdk_name``.

    Preserves the Pelgrom Vth mismatch budget by scaling ``W * L * N`` by
    ``(A_VT_target / A_VT_source) ** 2``. ``L`` is held fixed where
    possible; if the source ``L`` violates the target PDK's ``min_L``
    floor it is raised and ``W`` is reduced proportionally to keep the
    ``W * L`` product unchanged before applying the A_VT scaling. The
    finger count is held at the source value (and floored to >= 1).

    Raises
    ------
    ValueError
        If either PDK name is unknown, if W/L are non-positive, or if
        finger_count < 1.
    """
    if source_w_um <= 0.0:
        raise ValueError(f"source_w_um must be > 0, got {source_w_um}")
    if source_l_um <= 0.0:
        raise ValueError(f"source_l_um must be > 0, got {source_l_um}")
    if source_finger_count < 1:
        raise ValueError(
            f"source_finger_count must be >= 1, got {source_finger_count}"
        )

    source = _resolve_pdk(source_pdk_name)
    target = _resolve_pdk(target_pdk_name)

    notes: List[str] = []

    # Identity short-circuit when source == target (preserves sigma trivially).
    if source.name() == target.name():
        sigma_src = _sigma_proxy(
            source.avt_mv_um(device_type),
            source_w_um, source_l_um, source_finger_count,
        )
        return MigrationResult(
            source_pdk=source.name(),
            target_pdk=target.name(),
            source_w_um=source_w_um,
            source_l_um=source_l_um,
            source_finger_count=source_finger_count,
            target_w_um=source_w_um,
            target_l_um=source_l_um,
            target_finger_count=source_finger_count,
            sigma_preserved=True,
            notes=["identity migration (source PDK == target PDK)"],
        )

    avt_src_mv_um = source.avt_mv_um(device_type)
    avt_tgt_mv_um = target.avt_mv_um(device_type)

    # Step 1: choose L_target. Prefer source L; floor to target min_L.
    target_min_l_um = target.min_l_nm() / 1000.0
    l_target = source_l_um
    if l_target < target_min_l_um:
        l_target = target_min_l_um
        notes.append(
            f"min_L floor applied (source L={source_l_um:g} um -> "
            f"target L={l_target:g} um per {target.name()} min_L="
            f"{target_min_l_um:g} um)"
        )

    # Step 2: required (W*L*N)_target to preserve sigma.
    #   (W*L*N)_t >= r2 * (W*L*N)_s, where r2 = (A_VT_t / A_VT_s) ** 2.
    r2 = (avt_tgt_mv_um / avt_src_mv_um) ** 2
    required_wln = r2 * source_w_um * source_l_um * source_finger_count

    # Step 3: hold N fixed at source (>= 1 already enforced), solve for W.
    n_target = max(1, int(source_finger_count))
    w_target = required_wln / (l_target * n_target)

    # Step 4: sigma check (must hold by construction, but verify numerically).
    sigma_src = _sigma_proxy(
        avt_src_mv_um, source_w_um, source_l_um, source_finger_count
    )
    sigma_tgt = _sigma_proxy(avt_tgt_mv_um, w_target, l_target, n_target)
    # Allow tiny floating-point slack (1e-9 relative).
    sigma_preserved = sigma_tgt <= sigma_src * (1.0 + 1e-9)

    notes.append(
        f"A_VT ratio (target/source) = {avt_tgt_mv_um:g}/{avt_src_mv_um:g} "
        f"= {avt_tgt_mv_um / avt_src_mv_um:g}; W*L*N scaled by r^2={r2:g}"
    )

    return MigrationResult(
        source_pdk=source.name(),
        target_pdk=target.name(),
        source_w_um=source_w_um,
        source_l_um=source_l_um,
        source_finger_count=source_finger_count,
        target_w_um=w_target,
        target_l_um=l_target,
        target_finger_count=n_target,
        sigma_preserved=sigma_preserved,
        notes=notes,
    )
