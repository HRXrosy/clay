"""Replay content masters and their certified physical envelopes in one frame."""
from dataclasses import asdict
import json


def physical_master_boxes(blocks, projections=()):
    from clay.layout.place.constraints_spatial import FlexibleEnvelopeProjection, _flexible_seed_physical_views
    if type(projections) is not tuple or any(type(row) is not FlexibleEnvelopeProjection for row in projections):
        raise TypeError('physical master projections must be exact typed rows')
    result=dict(blocks)
    if not projections:
        return result
    physical,groups=_flexible_seed_physical_views(
        {name:box[:2] for name,box in blocks.items()},
        dims={name:(box[2]-box[0],box[3]-box[1]) for name,box in blocks.items()},
        constraints=tuple(row.constraint for row in projections))
    if any(row.bbox_nm!=groups[row.constraint.group] for row in projections):
        raise ValueError('physical master projection does not replay physical outline')
    result.update(physical)
    return result


def projection_payload(projections):
    return json.loads(json.dumps([asdict(row) for row in projections]))


def projections_from_payload(raw):
    from clay.layout.place.constraints_spatial import FlexibleEnvelopeConstraint, FlexibleEnvelopeProjection
    if type(raw) is not list:
        raise TypeError('physical master projection payload must be an exact list')
    def freeze(value):
        return tuple(freeze(item) for item in value) if type(value) is list else value
    rows=[]
    for row in raw:
        if type(row) is not dict or set(row)!={'constraint','bbox_nm','foreign_clearances_nm'}:
            raise ValueError('physical master projection payload fields differ')
        constraint=row['constraint']
        if type(constraint) is not dict:
            raise TypeError('physical master constraint payload must be an exact mapping')
        authority=FlexibleEnvelopeConstraint(**{key:freeze(value) for key,value in constraint.items()})
        rows.append(FlexibleEnvelopeProjection(authority,freeze(row['bbox_nm']),freeze(row['foreign_clearances_nm'])))
    return tuple(rows)
