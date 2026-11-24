"""
Abstract simulator driver.

Concrete drivers (NgspiceDriver, HspiceDriver, SpectreDriver) implement the
three required methods. The Clay verifier interacts only with this interface,
so adding a new simulator does not require changes to Clay's compiler or IR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Sequence

from ..ir import Testbench


class SimulatorDriver(ABC):
    """Triple of {emit, run, parse} that any simulator driver must provide."""

    name: str = "base"

    @abstractmethod
    def emit(self, tb: Testbench, out_path: Path) -> Path:
        """Write a simulator-specific netlist for the testbench.

        Returns the path to the emitted netlist (typically == out_path).
        """

    @abstractmethod
    def run(self, netlist_path: Path, work_dir: Path) -> Path:
        """Run the simulator in batch/silent mode on the netlist.

        Returns the path to the simulator log/output that contains
        measurement results.
        """

    @abstractmethod
    def parse_measures(
        self,
        log_path: Path,
        wanted: Sequence[str],
    ) -> Mapping[str, float]:
        """Extract measure-name → value from the simulator log.

        wanted: list of measure names declared in the testbench. Any extra
        measures the simulator computed are ignored.
        """

    # ─── Convenience: full pipeline ───────────────────────────────

    def emit_run_parse(
        self,
        tb: Testbench,
        work_dir: Path,
    ) -> tuple[Mapping[str, float], Path, Path]:
        """One-shot: emit netlist, run sim, parse results."""
        work_dir.mkdir(parents=True, exist_ok=True)
        netlist = self.emit(tb, work_dir / f"{tb.cell.name}.cir")
        log = self.run(netlist, work_dir)
        wanted = [m.name for m in tb.measures]
        measures = self.parse_measures(log, wanted)
        return measures, netlist, log
