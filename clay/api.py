"""Clay's single public layout-and-signoff facade (strangler-fig seam).

This module is the ONE public entrypoint a caller should reach for. It exists to
make the correct path the *only* importable one (pit-of-success): rather than
hand-assembling ``run_flow`` + an ad-hoc done-claim, a caller imports
:func:`layout_and_signoff` and gets back a typed :class:`LayoutResult`.

Strangler-fig contract (S1)
---------------------------
With ``certify=False`` (the default) this facade is a **byte-identical
pass-through** to :func:`clay.flow.engine.run_flow`: it forwards
``spec``/``out_dir``/``only``/``force``/``recorder`` VERBATIM and wraps the
returned :class:`~clay.flow.engine.FlowResult` unchanged in ``LayoutResult.flow``.
It switches NO router/placer and changes NO production output. The parity test
``tests/test_api_facade_parity.py`` measures
``run_flow(spec).to_dict() == layout_and_signoff(spec).flow.to_dict()`` on a real
spec.

The ``certify=True`` branch attaches the oracle-bound ``SignedOffLayout`` done-type
(a claim obtainable only through the independent KLayout LVS gate) via
:func:`clay.layout.route.signed_off.certify_signed_off`. ``certify`` defaults OFF,
so the live flow is byte-identical and untouched; certification is purely additive
(``LayoutResult.signed``). ``tech_id`` is REQUIRED when ``certify=True`` (the PDK whose
oracle certifies the layout) and ``full_extraction`` selects the PDK's full
device-extraction overlay; both are ignored by the verbatim pass-through.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from clay.flow.engine import FlowResult, run_flow

if False:  # pragma: no cover - typing only; the import is lazy at runtime (S2 path)
    from clay.layout.route.signed_off import SignedOffLayout, UnsignedLayout

__all__ = ["layout_and_signoff", "LayoutResult", "recover_precreate_primitive"]


def recover_precreate_primitive(spec: Any, *, project_root, task_id, preparation_manifest,
                                budget, cancellation=None):
    """Record a failed primitive stage after verifying a pre-create interruption."""
    from clay.flow.run_admission import recover_precreate_primitive as recover
    return recover(spec, project_root=project_root, task_id=task_id,
                   preparation_manifest=preparation_manifest, budget=budget,
                   cancellation=cancellation)


@dataclass(frozen=True)
class LayoutResult:
    """The typed result of a layout (+ optional signoff) run.

    ``flow`` is the unchanged :class:`~clay.flow.engine.FlowResult` from the live
    driver. ``signed`` is ``None`` for a non-certifying run (``certify=False``); on
    the ``certify=True`` path it carries the oracle-bound
    ``SignedOffLayout`` / ``UnsignedLayout`` union produced by
    :func:`clay.layout.route.signed_off.certify_signed_off`. Frozen so a result
    cannot be mutated into a stronger claim after the fact.
    """

    flow: FlowResult
    signed: Optional["Union[SignedOffLayout, UnsignedLayout]"] = None

    @property
    def passed(self) -> bool:
        """The flow's own verdict, surfaced for convenience. This is NOT a
        signoff claim -- a signoff claim is the ``signed`` done-type (S2)."""
        return self.flow.passed


def layout_and_signoff(
    spec: Any,
    *,
    out_dir: str | Path,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
    recorder: Any = None,
    budget: Any = None,
    cancellation: Any = None,
    project_root: str | Path | None = None,
    task_id: str | None = None,
    retry: Any = None,
    stage_retry: Any = None,
    stage_unwind_seconds: float | None = None,
    stage_role_trace_seconds: float | None = None,
    stage_role_trace_version: int | None = None,
    production_quiescence: Any = None,
    certify: bool = False,
    tech_id: Optional[str] = None,
    full_extraction: bool = False,
) -> LayoutResult:
    """Run Clay's layout flow for ``spec`` and return a typed :class:`LayoutResult`.

    Parameters
    ----------
    spec:
        The :class:`~clay.flow.spec.CircuitSpec` to lay out.
    out_dir, only, force, recorder:
        Forwarded VERBATIM to :func:`clay.flow.engine.run_flow` (see its
        docstring). With ``certify=False`` this facade adds nothing on top of the
        live driver.
    certify:
        ``False`` (default) -> byte-identical pass-through to ``run_flow``;
        ``signed`` is ``None``. ``True`` -> attach the oracle-bound
        ``SignedOffLayout`` / ``UnsignedLayout`` union (the independent KLayout LVS
        gate) as ``signed``; the wrapped ``flow`` is unchanged.
    tech_id, full_extraction:
        Consumed only by the ``certify=True`` certification path (``tech_id`` is
        REQUIRED there) and ignored by the verbatim pass-through.
    budget, cancellation, project_root, task_id, retry:
        Optional execution controls forwarded unchanged only when provided.
        They do not alter the wrapped flow result or physical certification.
    """
    if certify and only is not None:
        only = tuple(only)
        if 'primitive-observe' in only:
            raise ValueError('an observation-only flow cannot request physical certification')
    controls = {name: value for name, value in (
        ("budget", budget), ("cancellation", cancellation),
        ("project_root", project_root), ("task_id", task_id), ("retry", retry),
        ("stage_retry", stage_retry),
        ("stage_unwind_seconds", stage_unwind_seconds),
        ("stage_role_trace_seconds", stage_role_trace_seconds),
        ("stage_role_trace_version", stage_role_trace_version),
        ("production_quiescence", production_quiescence),
    ) if value is not None}
    flow: FlowResult = run_flow(
        spec,
        out_dir=out_dir,
        only=only,
        force=force,
        recorder=recorder,
        **controls,
    )

    if not certify:
        # Strangler-fig seam: verbatim pass-through, FlowResult wrapped unchanged.
        return LayoutResult(flow=flow, signed=None)

    if flow.verdict == 'unknown':
        raise ValueError('an observation-only flow cannot request physical certification')

    # certify=True -- attach the oracle-bound done-type. The independent KLayout LVS
    # oracle (clay.layout.route.signed_off.certify_signed_off) returns the typed
    # SignedOffLayout / UnsignedLayout union (NEVER raises on a non-match), which we
    # surface as LayoutResult.signed. The unchanged FlowResult is still wrapped in
    # ``flow`` -- certification is ADDITIVE, the live flow output is untouched.
    if tech_id is None:
        raise ValueError(
            "layout_and_signoff(certify=True) requires tech_id -- the PDK whose "
            "independent KLayout LVS oracle certifies the layout. Pass tech_id, or "
            "call with certify=False (the default) for the verbatim run_flow "
            "pass-through."
        )
    from clay.layout.route.signed_off import certify_signed_off

    signed = certify_signed_off(
        None, None, tech_id=tech_id, full_extraction=full_extraction, run_oracle=True
    )
    return LayoutResult(flow=flow, signed=signed)
