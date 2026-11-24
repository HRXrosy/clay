"""FullyDiffOpampWithCMFB — fully-differential single-stage opamp
with a dedicated CMFB control node driving the NMOS active-load
gates through an internal inverter.

This is the v0.0.84 closure of the v0.0.82 / v0.0.83 honest-scope
deferred finding: the v0.0.77 fully-diff testbench's CMFB regulator
output (top-level ``vcmfb``) had no DUT-side handle. Without one, the
CMFB loop closed only through the 1 MΩ ``Rcm1`` / ``Rcm2`` sensor-
leakage path, leaving ``cmfb_cm_offset`` at a device-determined value
(~−900 mV on the v0.0.82 synthetic FD pair) regardless of
``cmfb_ea_gain``.

This cell exposes the standard 6-port FD convention
``[inn, inp, outn, outp, vdd, vss]`` but routes the NMOS load gates
through an internal CMOS-inverter stage whose input is the global
``vcmfb`` net. Sweeping ``cmfb_ea_gain`` while running through the
v0.0.77 / 78 / 79 / 82 fully-diff testbench family produces the
expected monotonic ``|cmfb_cm_offset|`` increase as gain drops — the
operational definition of finite-EA-gain CMFB degradation.

Production callers should use ``simulate_with_cmfb_loop`` (in this
module) to drive the cell through the existing FD testbench family.
That wrapper handles the ``.global vcmfb`` injection required to
bridge the testbench's top-level ``vcmfb`` net (driven by ``E_cmfb``)
to the cell's internal ``vcmfb`` references; without it the names
collide and the loop stays open.

See ``clay/templates/fully_differential_with_cmfb.py`` for the full
topology description, bias plan, and loop-polarity derivation.
"""

from __future__ import annotations

from typing import Any, Optional

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.fully_differential_with_cmfb import (
    build_fully_diff_opamp_with_cmfb,
)


@register_cell("FullyDiffOpampWithCMFB")
def FullyDiffOpampWithCMFB(spec: Spec, **kwargs: Any) -> Cell:
    """Build a fully-differential single-stage opamp Cell IR with
    CMFB-controlled NMOS loads.

    kwargs are forwarded to the template builder for sizing
    overrides: ``I_ref``, ``W_bp``, ``L_bp``, ``N_tail``, ``W_diff``,
    ``L_diff``, ``W_load``, ``L_load``, ``W_inv_n``, ``L_inv_n``,
    ``W_inv_p``, ``L_inv_p``.

    The returned Cell has 6 external ports
    ``[inn, inp, outn, outp, vdd, vss]``. The CMFB control node
    ``vcmfb`` is intentionally NOT exposed as a port — it must be a
    global net at simulation time so the standard 6-port FD testbench
    family (which hard-codes a 6-port xDUT instantiation) routes here
    via ``_detect_port_shape``. Use ``simulate_with_cmfb_loop`` to
    handle the ``.global vcmfb`` injection automatically.
    """
    return build_fully_diff_opamp_with_cmfb(
        cell_name=kwargs.pop("cell_name", "FullyDiffOpampWithCMFB"),
        spec=spec,
        **kwargs,
    )


# ─── Handle-side helper: wrap export to inject ``.global vcmfb`` ───
#
# The v0.0.77 fully-differential testbench builders (AC, tran,
# dc_sweep) instantiate the DUT as
# ``xDUT inn inp outn outp vdd vss <cell>`` — six ports, hard-coded.
# A 7th external port for ``vcmfb`` would either break that
# instantiation line (port-count mismatch) or require editing the
# bridge (out of scope for this mission slice).
#
# Solution: make ``vcmfb`` a global net via ``.global vcmfb`` in the
# exported netlist. ngspice promotes references to the same name
# inside any subckt to that single top-level net, so the CMFB error
# amp's output (top-level ``vcmfb`` driven by ``E_cmfb``) connects
# directly to the DUT's internal CMFB receiver (the inverter input).
#
# ``_GlobalNetHandleWrapper`` is a Python proxy around a
# ``CellGraphHandle`` that delegates every method but
# ``export_spice_subckt``, which it overrides to prepend the
# ``.global vcmfb`` directive immediately before the ``.subckt``
# header. The shape detector (``_detect_port_shape``) reads
# ``.subckt`` to recover port names — the prepended ``.global`` line
# does NOT change the port list. The dispatch contract stays intact.


class _GlobalNetHandleWrapper:
    """Proxy wrapper around a ``CellGraphHandle`` that injects
    ``.global <net>`` directives into the exported subckt text.

    Used to bridge top-level testbench nets (e.g. ``vcmfb`` produced
    by the v0.0.77 CMFB regulator) into a 6-port FD DUT cell that
    references the same name internally. Without the ``.global``
    directive the names would collide as two separate nets — the
    top-level ``vcmfb`` (E_cmfb output, top-level scope) and the
    subckt-local ``xdut.vcmfb`` (load-gate connection, subckt-local
    scope).

    This wrapper preserves every other ``CellGraphHandle`` method
    (``pdk_id``, ``cell_name``, etc.) by delegation, so it drops
    transparently into ``simulate_hypergraph`` without touching the
    bridge dispatch logic.
    """

    def __init__(self, inner, globals_: tuple[str, ...]):
        """Wrap ``inner`` (a CellGraphHandle) and declare ``globals_``
        as global nets in the exported subckt text.

        Args:
            inner: the wrapped CellGraphHandle. Method calls fall
                through except ``export_spice_subckt``.
            globals_: tuple of net names to be marked as global. Each
                name produces one ``.global <name>`` line in the
                emitted netlist, placed immediately before the
                ``.subckt`` header so ngspice sees it before parsing
                the subckt body.
        """
        self._inner = inner
        self._globals = tuple(globals_)

    def __getattr__(self, name):
        # Delegate every attribute access except ``export_spice_subckt``
        # to the inner handle. This includes ``pdk_id``, ``cell_name``,
        # ``external_port_count``, and any other PyO3 method the
        # bridge calls during dispatch.
        return getattr(self._inner, name)

    def export_spice_subckt(self, bindings):
        """Emit the wrapped handle's subckt text with ``.global <net>``
        lines inserted immediately before the ``.subckt`` header.

        The shape detector reads the first ``.subckt`` line in the
        emitted text to recover external port names — prepending
        ``.global`` lines does not change the port-name list, so the
        v0.0.75 / 77 dispatch chain routes unchanged.
        """
        raw = self._inner.export_spice_subckt(bindings)
        if not self._globals:
            return raw
        lines = raw.splitlines()
        out: list[str] = []
        inserted = False
        for line in lines:
            if line.lstrip().startswith(".subckt") and not inserted:
                for net in self._globals:
                    out.append(f".global {net}")
                inserted = True
            out.append(line)
        if not inserted:
            # Defensive: no .subckt found (degenerate handle). Prepend
            # the .global lines at the top so the netlist is still
            # syntactically correct.
            for net in self._globals:
                out.insert(0, f".global {net}")
        return "\n".join(out)


def wrap_handle_for_cmfb_loop(handle, *, extra_globals: tuple[str, ...] = ()):
    """Return a ``CellGraphHandle``-like wrapper that injects
    ``.global vcmfb`` (plus any extras) into the exported subckt text.

    Use this when calling ``simulate_hypergraph(...)`` on a fully-
    differential opamp cell that wants the v0.0.77 testbench's CMFB
    regulator output (``vcmfb``) to drive an internal CMFB-receiving
    node. Without the wrapper the cell's internal ``vcmfb`` references
    are subckt-local and the loop stays open through the resistor
    leakage path only.

    Args:
        handle: a CellGraphHandle returned by ``cell_to_hypergraph``
            or ``CellGraphHandle.new_empty(...)``.
        extra_globals: additional net names to mark global. ``vcmfb``
            is always included; pass a tuple of strings to add more
            (e.g. for future templates that need a CMFB reference
            voltage as a global too).

    Returns:
        A duck-typed handle wrapper that ``simulate_hypergraph``
        accepts unchanged.
    """
    globals_ = ("vcmfb",) + tuple(extra_globals)
    return _GlobalNetHandleWrapper(handle, globals_)


def simulate_with_cmfb_loop(
    handle,
    *,
    testbench_kind: str = "dc_sweep",
    extra_globals: tuple[str, ...] = (),
    **simulate_kwargs,
) -> dict[str, float]:
    """Convenience wrapper that calls ``simulate_hypergraph`` on a
    fully-differential opamp cell with the ``.global vcmfb`` directive
    automatically injected.

    Args:
        handle: a CellGraphHandle (e.g. from
            ``cell_to_hypergraph(FullyDiffOpampWithCMFB(spec))``).
        testbench_kind: one of ``"ac"`` / ``"tran"`` / ``"dc_sweep"``.
            Default ``"dc_sweep"`` — the headline path that
            demonstrates the gain-dependence of ``cmfb_cm_offset``
            unambiguously (the AC builder's ``cmfb_cm_offset`` .meas
            measures the AC response rather than the DC operating
            point, which is balanced and cancels — that's a v0.0.82
            measure-line limitation, not a topology issue with this
            cell).
        extra_globals: forwarded to ``wrap_handle_for_cmfb_loop``.
        **simulate_kwargs: forwarded to ``simulate_hypergraph``.
            Notable: ``cmfb_ea_gain`` (default 1e6),
            ``cmfb_ea_bw`` (default inf), ``corner``, ``supply_v``.

    Returns:
        The same dict that ``simulate_hypergraph`` returns. On the
        DC-sweep path with the default ideal EA, expect
        ``cmfb_cm_offset ≈ 0``; finite EA gain leaves a residual
        scaling as roughly ``vcm_target / cmfb_ea_gain``.
    """
    # Local import: clay._ir_bridge imports clay_ir lazily, so this
    # avoids a hard dep at package import time. The skipif guards in
    # the tests handle the no-clay_ir case.
    from ..._ir_bridge import simulate_hypergraph

    wrapped = wrap_handle_for_cmfb_loop(handle, extra_globals=extra_globals)
    return simulate_hypergraph(
        wrapped, testbench_kind=testbench_kind, **simulate_kwargs,
    )
