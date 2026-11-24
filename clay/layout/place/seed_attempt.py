"""Separate a restricted band seed attempt from the complete placement proof.

Band decomposition owns half of the caller's native solve budget. Failure in
that restricted model cannot prove the original model infeasible. Only a typed
local infeasibility can discard the seed; cancellation and implementation
errors remain terminal. Inapplicable decompositions are separately recorded;
they establish no infeasibility. The complete master owns the reserved half.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math

from clay.flow.execution_session import current_session
from clay.layout.placement_infeasibility import PlacementInfeasibilityError, SufficientCoreDiagnostic


class BandedSeedNotApplicable(ValueError):
    """Valid complete constraints are outside the independent-band strategy."""


@dataclass(frozen=True, slots=True)
class BandedSeedAttempt:
    positions: dict[str, tuple[int, int]] | None
    remaining_budget: float
    rejection: SufficientCoreDiagnostic | None = None
    inapplicability: str | None = None

    def source_kwargs(self, original):
        if self.positions is None:
            return original
        return {'source_seed_positions': self.positions, 'certified_source_seed': True}


def attempt_banded_seed(builder, *args, max_time_s, **kwargs):
    """Run the heuristic and retain its restricted-domain diagnostic if rejected."""
    if isinstance(max_time_s, bool) or not math.isfinite(max_time_s) or max_time_s < 0:
        raise ValueError('band seed budget must be finite and non-negative')
    reserved = float(max_time_s) / 2
    session = current_session()
    phase = (session.phase('placement.seed.banded_attempt', details={
        'native_seed_budget': reserved, 'native_master_budget': reserved,
        'proof_scope': 'restricted_band_seed',
    }) if session else nullcontext())
    with phase as handle:
        try:
            positions, remaining = builder(*args, max_time_s=max_time_s, **kwargs)
        except BandedSeedNotApplicable as exc:
            if handle:
                handle.annotate(seed_status='not_applicable', seed_applicability={
                    'scope': 'restricted_band_seed', 'reason': str(exc)})
            return BandedSeedAttempt(None, reserved, inapplicability=str(exc))
        except PlacementInfeasibilityError as exc:
            if handle:
                handle.annotate(seed_status='rejected', seed_rejection=exc.diagnostic.canonical_data())
            return BandedSeedAttempt(None, reserved, exc.diagnostic)
        if remaining != reserved:
            raise RuntimeError('band decomposition changed the reserved master budget')
        if handle:
            handle.annotate(seed_status='available')
        return BandedSeedAttempt(positions, reserved)
