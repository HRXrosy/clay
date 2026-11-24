"""CapabilityMatrix: probe-based support status for layout intent primitives per PDK.

Per the architecture's "all incompleteness explicit" rule, a primitive that Clay cannot
build on a given PDK must be NAMED ``unsupported`` -- never silently substituted (e.g.
the gf180 poly resistor collapsed to a wire). Each capability here is PROBED: the matrix
actually invokes the real builder on the target PDK and reports

* ``PASS`` (with the probe evidence) when the primitive builds;
* ``UNSUPPORTED`` (with a reason) when the builder has no facts / no spec for that PDK;
* ``ERROR`` when the probe raised unexpectedly.

So the matrix cannot claim a capability it does not have -- it is honest by construction.
Verdicts are :class:`clay.signoff_verdict.CheckerVerdict`, so a capability gap threads
straight into the obligation/blocking machinery (a REQUIRED unsupported primitive blocks
sign-off; an optional one is tolerated).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from clay.core_contract import VerificationStatus
from clay.signoff_verdict import CheckerVerdict

__all__ = ["CAPABILITIES", "capability_status", "capability_matrix"]


# --------------------------------------------------------------------------- #
# probes: each returns (status, evidence, reason). A probe NEVER fabricates a
# pass -- it builds the real primitive (or proves the facts/spec are absent).
# --------------------------------------------------------------------------- #
def _probe_poly_res(tech_id: str) -> Tuple[VerificationStatus, Dict[str, Any], str]:
    from clay.layout.passive_cell import build_poly_res
    from clay.layout.passive_device_cell import build_resistor
    from clay.layout.tech import LayoutTech

    try:
        tech = LayoutTech.resolve(tech_id)
    except (KeyError, ValueError) as exc:
        return VerificationStatus.UNSUPPORTED, {"pdk": tech_id}, (
            f"layout technology is unavailable for {tech_id!r}: {exc}"
        )
    except Exception as exc:  # pragma: no cover - unexpected resolver failure
        return VerificationStatus.ERROR, {"pdk": tech_id}, (
            f"LayoutTech.resolve: {type(exc).__name__}: {exc}"
        )
    attempts = []
    cell = None
    materializer = ""
    for name, builder in (
        ("passive_cell.build_poly_res", lambda: build_poly_res(tech_id, 1000.0)),
        ("passive_device_cell.build_resistor", lambda: build_resistor(tech, 1000.0)),
    ):
        try:
            cell = builder()
            materializer = name
            break
        except (KeyError, ValueError) as exc:
            attempts.append(f"{name}: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected builder failure
            return VerificationStatus.ERROR, {"pdk": tech_id}, (
                f"{name}: {type(exc).__name__}: {exc}"
            )
    if cell is None:
        return VerificationStatus.UNSUPPORTED, {"pdk": tech_id}, (
            f"poly-resistor materializers are unavailable for {tech_id!r}: "
            + "; ".join(attempts)
        )
    layer_map = getattr(cell, "polys_by_layer", {}) or {}
    layers = sorted({
        (int(poly.layer), int(poly.datatype))
        for poly in tuple(getattr(cell, "polygons", ()) or ())
    } | {
        (int(key[0]), int(key[1])) for key in layer_map
    })
    terminals = len(getattr(cell, "terminals", {}) or {})
    if not layers or (
        materializer == "passive_device_cell.build_resistor" and terminals != 2
    ):
        return VerificationStatus.ERROR, {
            "pdk": tech_id,
            "layers": layers,
            "terminals": terminals,
        }, "canonical poly-resistor materializer returned an incomplete cell"
    return VerificationStatus.PASS, {
        "pdk": tech_id,
        "layers": layers,
        "terminals": terminals,
        "source": getattr(cell, "source", ""),
        "materializer": materializer,
    }, ""


def _probe_mim_cap(tech_id: str) -> Tuple[VerificationStatus, Dict[str, Any], str]:
    from clay.layout.passive_device_cell import build_capacitor_for_tech
    from clay.layout.route.pins import _mim_spec_for
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    spec = _mim_spec_for(tech)
    if spec is None:
        return VerificationStatus.UNSUPPORTED, {"pdk": tech_id}, (
            f"no MIM device-recognition spec for {tech_id!r} (cap cannot be wired/extracted)"
        )
    try:
        cell = build_capacitor_for_tech(tech, 1.0e-12, nets={"p": "a", "n": "b"})
    except Exception as e:  # pragma: no cover - unexpected builder failure
        return VerificationStatus.ERROR, {"pdk": tech_id}, f"{type(e).__name__}: {e}"
    nterms = len(getattr(cell, "terminals", {}) or {})
    return VerificationStatus.PASS, {"pdk": tech_id, "terminals": nterms,
                                     "has_mim_spec": True}, ""


# the registered layout intent primitives. Extend by adding a probe here -- the matrix
# then reports it for every PDK (codex: new capability = new probe, not an `if`).
CAPABILITIES: Dict[str, Callable[[str], Tuple[VerificationStatus, Dict[str, Any], str]]] = {
    "poly_res": _probe_poly_res,
    "mim_cap": _probe_mim_cap,
}


def capability_status(capability: str, tech_id: str, *, required: bool = True) -> CheckerVerdict:
    """Probe ``capability`` on ``tech_id`` and return the verdict. An unregistered
    capability is itself UNSUPPORTED (never a crash)."""
    probe = CAPABILITIES.get(capability)
    obligation_id = f"capability.{capability}.{tech_id}"
    if probe is None:
        return CheckerVerdict(
            checker="capability", obligation_id=obligation_id,
            status=VerificationStatus.UNSUPPORTED,
            evidence={"capability": capability, "pdk": tech_id},
            blocker_reason=f"no probe registered for capability {capability!r}",
            required=required,
        )
    status, evidence, reason = probe(tech_id)
    if status == VerificationStatus.PASS and not evidence:
        evidence = {"capability": capability, "pdk": tech_id}
    return CheckerVerdict(
        checker="capability", obligation_id=obligation_id, status=status,
        evidence=evidence, blocker_reason=reason, required=required,
    )


def capability_matrix(tech_id: str, *, required: bool = True) -> Dict[str, CheckerVerdict]:
    """Every registered capability's verdict on ``tech_id`` (none silently skipped)."""
    return {cap: capability_status(cap, tech_id, required=required) for cap in CAPABILITIES}
