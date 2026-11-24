"""Actual placement evidence for a complete source-bound non-MOS assembly.

The captured member boxes remain the density denominator. Only a rectangle
collision view substitutes one external box per assembly. Capture replays all
member artwork against the native binding; this is no general overlap waiver.
"""
from dataclasses import dataclass
import hashlib
import json

from clay.layout.source_bound_nonmos_plan import SourceBoundNonmosMacroPlan


_SEAL = object()
_MAX_MEMBERS = 256


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _plans(value):
    if type(value) is not tuple or not 0 < len(value) <= _MAX_MEMBERS:
        raise TypeError('nonmos placement plans require a nonempty bounded exact tuple')
    if any(type(plan) is not SourceBoundNonmosMacroPlan for plan in value):
        raise TypeError('nonmos placement requires exact typed source-bound plans')
    ids = tuple(plan.macro_id for plan in value)
    owners = tuple(owner for plan in value for owner in plan.members)
    if ids != tuple(sorted(set(ids))) or len(owners) != len(set(owners)):
        raise ValueError('nonmos placement plans must be canonical and owner-disjoint')
    if len(owners) > _MAX_MEMBERS:
        raise ValueError('nonmos placement member denominator exceeds the physical cap')
    for plan in value:
        plan.__post_init__()
    return value


def _blocks(value):
    if type(value) is not tuple or len(value) > _MAX_MEMBERS:
        raise TypeError('nonmos member blocks require a bounded exact tuple')
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError('nonmos member block must be an exact pair')
        name, box = item
        if type(name) is not str or not name or type(box) is not tuple or len(box) != 4:
            raise ValueError('nonmos member block requires a name and exact bbox')
        if any(type(v) is not int for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('nonmos member bbox requires positive integer-nm extent')
    if value != tuple(sorted(value)) or len({name for name, _ in value}) != len(value):
        raise ValueError('nonmos member blocks must have canonical unique names')
    return value


@dataclass(frozen=True, slots=True, init=False)
class NonmosAssemblyPlacementBinding:
    plans: tuple[SourceBoundNonmosMacroPlan, ...]
    plan_sha256s: tuple[tuple[str, str], ...]
    translations_nm: tuple[tuple[str, tuple[int, int]], ...]
    member_bboxes_nm: tuple
    macro_bboxes_nm: tuple
    _captured_sha256: str
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('use capture_nonmos_assembly_placement with current actual artwork')

    def _payload(self):
        return {'schema': 'clay.nonmos-assembly-placement/v1',
                'plans': self.plan_sha256s, 'translations_nm': self.translations_nm,
                'member_bboxes_nm': self.member_bboxes_nm, 'macro_bboxes_nm': self.macro_bboxes_nm}

    def validate(self):
        if self._seal is not _SEAL:
            raise ValueError('nonmos placement binding was not captured')
        plans = _plans(self.plans)
        expected_hashes = tuple((plan.macro_id, plan.sha256) for plan in plans)
        if self.plan_sha256s != expected_hashes:
            raise ValueError('nonmos placement captured plan identity changed')
        if type(self.translations_nm) is not tuple or len(self.translations_nm) != len(plans):
            raise ValueError('nonmos placement binding requires every plan translation')
        expected_members, expected_macros = [], []
        for plan, item in zip(plans, self.translations_nm):
            if (type(item) is not tuple or len(item) != 2 or item[0] != plan.macro_id
                    or type(item[1]) is not tuple or len(item[1]) != 2
                    or any(type(v) is not int for v in item[1])):
                raise ValueError('nonmos placement binding translation is malformed')
            dx, dy = item[1]
            offset = (dx, dy, dx, dy)
            expected_macros.append(('nonmos:'+plan.macro_id,
                tuple(v+d for v, d in zip(plan.local_bbox_nm, offset))))
            expected_members.extend((member.member,
                tuple(v+d for v, d in zip(member.local_bbox_nm, offset))) for member in plan.member_plans)
        if (_blocks(self.member_bboxes_nm) != tuple(sorted(expected_members))
                or _blocks(self.macro_bboxes_nm) != tuple(sorted(expected_macros))):
            raise ValueError('nonmos placement binding differs from complete captured member bboxes')
        if _digest(self._payload()) != self._captured_sha256:
            raise ValueError('nonmos placement captured binding identity changed')
        return self

    @property
    def sha256(self):
        self.validate()
        return self._captured_sha256

    def payload(self):
        self.validate()
        return self._payload()

    def require_free_device_blocks(self, free_device_blocks):
        self.validate()
        blocks = dict(_blocks(free_device_blocks))
        for name, bbox in self.member_bboxes_nm:
            if blocks.get(name) != bbox:
                raise ValueError('nonmos placement member bbox differs from actual capture: '+name)

    def collision_blocks(self, free_device_blocks):
        self.require_free_device_blocks(free_device_blocks)
        members = {name for name, _ in self.member_bboxes_nm}
        external = tuple(item for item in free_device_blocks if item[0] not in members)
        if {name for name, _ in free_device_blocks} & {name for name, _ in self.macro_bboxes_nm}:
            raise ValueError('nonmos collision macro name conflicts with a free member')
        return tuple(sorted((*external, *self.macro_bboxes_nm)))


def capture_nonmos_assembly_placement(plans, *, devices):
    """Recheck full current polygons; retain immutable boxes and plan identities."""
    from collections.abc import Mapping
    from clay.layout.nonmos_local_assembly import checkpoint
    plans = _plans(plans)
    if not isinstance(devices, Mapping):
        raise TypeError('nonmos placement requires an actual device mapping')
    translations, members, macros = [], [], []
    for plan in plans:
        checkpoint()
        dx, dy = plan.translation_nm(devices)
        translations.append((plan.macro_id, (dx, dy)))
        members.extend((owner, devices[owner].bbox) for owner in plan.members)
        macros.append(('nonmos:'+plan.macro_id,
            tuple(v+d for v, d in zip(plan.local_bbox_nm, (dx, dy, dx, dy)))))
    result = object.__new__(NonmosAssemblyPlacementBinding)
    for name, value in (('plans', plans), ('plan_sha256s', tuple((p.macro_id, p.sha256) for p in plans)),
                        ('translations_nm', tuple(translations)), ('member_bboxes_nm', tuple(sorted(members))),
                        ('macro_bboxes_nm', tuple(sorted(macros))), ('_seal', _SEAL)):
        object.__setattr__(result, name, value)
    object.__setattr__(result, '_captured_sha256', _digest(result._payload()))
    return result.validate()
