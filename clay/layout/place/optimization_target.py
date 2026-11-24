"""Explicit stopping targets for optional placement improvement.

A target never admits a design or relaxes a hard constraint. Callers still
materialize and replay the final geometry, certify it, and apply acceptance
policy. Generated-incumbent proof replay keeps its original proof path.
The supported host is the banded area master: the target controls its optional
frontier/full2d refinement. Native primary and soft-position solves keep their
existing budgets and order. Other public master modes reject this target.
"""
from dataclasses import dataclass

from clay.flow.execution_session import current_session


@dataclass(frozen=True, slots=True)
class AreaTarget:
    maximum_area_nm2: int

    def __post_init__(self):
        if type(self.maximum_area_nm2) is not int or self.maximum_area_nm2 <= 0:
            raise ValueError('area target must be an exact positive integer nm2')


def validate_area_target(target):
    if target is not None and type(target) is not AreaTarget:
        raise TypeError('area target must be one immutable AreaTarget or None')
    if target is not None:
        target.__post_init__()


def area_target_reached(target, *, area_nm2, hard_replayed, incumbent_replay=False):
    validate_area_target(target)
    if type(hard_replayed) is not bool or type(incumbent_replay) is not bool:
        raise TypeError('area target replay flags must be exact booleans')
    if target is None or not hard_replayed or incumbent_replay:
        return False
    if type(area_nm2) is not int or area_nm2 <= 0:
        raise ValueError('replayed physical area must be a positive integer nm2')
    reached = area_nm2 <= target.maximum_area_nm2
    session = current_session()
    if session:
        with session.phase('placement.optimization.target', details={
            'physical_area_nm2': area_nm2, 'maximum_area_nm2': target.maximum_area_nm2,
            'hard_replayed': True, 'target_reached': reached,
            'scope': 'optional_improvement_only', 'final_geometry_replay': 'still_required',
        }):
            pass
    return reached


def refine_below_area_target(refine, positions, *, target, area_nm2,
                            hard_replayed, incumbent_replay=False, **kwargs):
    if area_target_reached(target, area_nm2=area_nm2, hard_replayed=hard_replayed,
                           incumbent_replay=incumbent_replay):
        return dict(positions), None, False
    return refine(positions, **kwargs)
