"""Detached value snapshots of the operands read by joint row validation.

Snapshots provide exact within-invocation mutation detection, not validation or
source authority. No property payload, geometry extractor or cache is invoked.
"""
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import math
from clay.layout.backend import Point
from clay.layout.place.row_detail_timing import measured as _row_measure

_UNSET = object()
_DATACLASS_FIELDS_CACHE: dict[type, tuple[str, ...] | None] = {}


def _dataclass_fields_or_none(kind: type) -> tuple[str, ...] | None:
    names = _DATACLASS_FIELDS_CACHE.get(kind, _UNSET)
    if names is _UNSET:
        names = tuple(field.name for field in fields(kind)) if is_dataclass(kind) else None
        _DATACLASS_FIELDS_CACHE[kind] = names
    return names


def freeze_row_values(value):
    """Copy a bounded primitive/dataclass tree, including type and multiplicity.

The memo only shares repeated subtrees during this one traversal. Nothing from
a previous comparison survives, and mutable input objects never enter output.
"""
    from clay.layout.joint_row_geometry import joint_checkpoint
    check = joint_checkpoint()
    memo, active, count = {}, {}, 0
    def freeze(item, depth=0):
        nonlocal count
        count += 1
        if count % 2048 == 0:
            check()
        if count > 8_000_000 or depth > 64:
            raise ValueError('row replay operand snapshot exceeds its bounded shape')
        kind = type(item)
        if item is None or kind is int or kind is str or kind is bool or kind is bytes:
            return kind, item
        if kind is tuple and depth + 1 <= 64:
            n_items = len(item)
            if n_items == 0:
                return tuple, ()
            if n_items == 2 and type(item[0]) is int and type(item[1]) is int:
                count += 2
                if count > 8_000_000:
                    raise ValueError('row replay operand snapshot exceeds its bounded shape')
                return tuple, ((int, item[0]), (int, item[1]))
            if (
                n_items == 4
                and type(item[0]) is int
                and type(item[1]) is int
                and type(item[2]) is int
                and type(item[3]) is int
            ):
                count += 4
                if count > 8_000_000:
                    raise ValueError('row replay operand snapshot exceeds its bounded shape')
                return tuple, ((int, item[0]), (int, item[1]), (int, item[2]), (int, item[3]))
        if kind is Point and depth + 1 <= 64:
            pt_dict = item.__dict__
            if len(pt_dict) == 2:
                px = pt_dict.get('x')
                py = pt_dict.get('y')
                if type(px) is int and type(py) is int:
                    count += 2
                    if count > 8_000_000:
                        raise ValueError('row replay operand snapshot exceeds its bounded shape')
                    return Point, (('x', (int, px)), ('y', (int, py)))
        if kind is float:
            if not math.isfinite(item):
                raise ValueError('row replay snapshot contains nonfinite geometry')
            return kind, item.hex()
        if kind is Decimal:
            if not item.is_finite():
                raise ValueError('row replay snapshot contains a nonfinite source value')
            return kind, tuple(item.as_tuple())
        if kind is Fraction:
            return kind, item.numerator, item.denominator
        if isinstance(item, Enum):
            return kind, freeze(item.value, depth+1)
        identity = id(item)
        if identity in active:
            # Parsed CellDef -> ClayFile -> cell_defs legitimately points back
            # into this graph. Preserve the link using this traversal's number;
            # never retain the mutable object or omit the parent source fields.
            return kind, 'active_back_reference', active[identity]
        if identity in memo:
            return memo[identity]
        active[identity] = count
        if kind is tuple or kind is list:
            result = kind, tuple([freeze(part, depth+1) for part in item])
        elif kind is dict:
            if all(type(key) is str for key in item):
                result = kind, tuple([((str, key), freeze(item[key], depth+1)) for key in sorted(item)])
            elif all(type(key) is int for key in item):
                result = kind, tuple([((int, key), freeze(item[key], depth+1)) for key in sorted(item)])
            else:
                if any(type(key) not in (str, int, tuple) for key in item):
                    raise ValueError('row replay snapshot mapping has unsupported keys')
                result = kind, tuple(sorted([(freeze(key, depth+1), freeze(part, depth+1))
                    for key, part in item.items()], key=lambda pair: repr(pair[0])))
        else:
            names = None if isinstance(item, type) else _dataclass_fields_or_none(kind)
            if names is not None:
                item_dict = getattr(item, '__dict__', None)
                if not item_dict:
                    result = kind, tuple([(name, freeze(getattr(item, name), depth+1))
                        for name in names])
                elif len(item_dict) == len(names):
                    try:
                        result = kind, tuple([(name, freeze(item_dict[name], depth+1))
                            for name in names])
                    except KeyError:
                        extra_names = sorted(set(item_dict) - set(names))
                        all_names = (*names, *extra_names) if extra_names else names
                        result = kind, tuple([(name, freeze(getattr(item, name), depth+1))
                            for name in all_names])
                else:
                    extra_names = sorted(set(item_dict) - set(names))
                    all_names = (*names, *extra_names) if extra_names else names
                    result = kind, tuple([(name, freeze(getattr(item, name), depth+1))
                        for name in all_names])
            else:
                raise TypeError('row replay snapshot has unsupported operand type '+kind.__name__)
        del active[identity]
        memo[identity] = result
        return result
    result = freeze(value)
    check()
    return result


def geometry_snapshot(geometry, members):
    """Include all native fields and the actual derived extent/member pin view."""
    from clay.layout.common_centroid_array import CommonCentroidArray
    from clay.layout.mosaic_geometry import MosaicGeometry
    if type(geometry) not in (CommonCentroidArray, MosaicGeometry):
        raise TypeError('row replay snapshot requires a supported exact native geometry')
    return (
        type(geometry),
        freeze_row_values(
            (
                vars(geometry),
                tuple(geometry.extent()),
                tuple((name, geometry.member_pins(name)) for name in members),
            )
        ),
    )


def materialization_snapshot(materialization):
    """Certificate/placement frames without duplicating the per-entry artwork."""
    return freeze_row_values((materialization.certificate, materialization.residuals,
        tuple(tuple((field.name, getattr(row, field.name)) for field in fields(row)
            if field.name != 'geometry') for row in materialization.realizations)))


def source_projection_snapshot(projection, *, validated_native_program=None,
                               computed_geometry_snapshots=None):
    # Dataclass fields preserve raw replay inputs, including full unconsumed
    # program members; unlike payload(), this never rebuilds an evidence tree.
    if projection is not None and getattr(projection, 'native_witness', None) is not None:
        from clay.layout.native_diffusion_source import (require_native_pcell_witness,
            _require_native_pcell_witness_operands)

        witness = projection.native_witness
        if validated_native_program is None:
            _row_measure('projection.witness', require_native_pcell_witness, witness, cell=witness.cell, tech=witness.tech,
                                        overlay=projection.reference_overlay,
                                        source_cir=projection.source_cir)
            return (_row_measure('projection.fields', freeze_row_values, {name: value for name,value in vars(projection).items()
                                       if name != 'native_witness'}),
                    id(witness), id(witness.bundle), witness.sha256)
        else:
            # A complete native witness was already replayed by the live scope.
            # Snapshot every operand that replay consumed; no value, geometry,
            # producer file, source artifact or template output is trusted by hash
            # from the previous check without rereading its current bytes.
            import hashlib
            import json
            from pathlib import Path
            _require_native_pcell_witness_operands(witness,cell=witness.cell,tech=witness.tech,
                overlay=projection.reference_overlay,source_cir=projection.source_cir)
            entries={entry.group:entry for entry in validated_native_program.entries}
            if len(entries)!=len(validated_native_program.entries):
                raise ValueError('row replay native source groups are duplicated')
            rows=[]
            for group,geometry,folder in witness.rows:
                entry=entries.get(group)
                if entry is None:
                    raise ValueError('row replay native witness owner is missing')
                folder=Path(folder)
                receipt_path=folder/'template-receipt.json'
                receipt_bytes=receipt_path.read_bytes()
                receipt=json.loads(receipt_bytes)
                outputs=receipt.get('outputs',{})
                if type(outputs) is not dict:
                    raise ValueError('row replay native template outputs are malformed')
                files=tuple((name,hashlib.sha256((folder/name).read_bytes()).hexdigest())
                            for name in sorted(outputs))
                if any(type(outputs[name]) is not str or outputs[name] != digest for name,digest in files):
                    raise ValueError('row replay native template output differs from its receipt')
                geom_snap=geometry_snapshot(geometry,entry.consumed_devices)
                if computed_geometry_snapshots is not None:
                    computed_geometry_snapshots[id(geometry)]=(tuple(entry.consumed_devices),geom_snap)
                rows.append((group,id(geometry),str(folder),geom_snap,
                             hashlib.sha256(receipt_bytes).hexdigest(),files))
            bundle=witness.bundle
            proj_fields={name: value for name,value in vars(projection).items()
                         if name != 'native_witness'}
            combined_frozen=freeze_row_values((
                proj_fields,
                vars(witness.cell),
                witness.overlay,
                witness.row_candidates if witness.row_candidates else (),
            ))
            native_operands=(
                tuple(rows),
                hashlib.sha256((bundle.output_dir/'geometry.json').read_bytes()).hexdigest(),
                hashlib.sha256((bundle.output_dir/'command.json').read_bytes()).hexdigest(),
            )
        return (combined_frozen,
                id(witness), id(witness.bundle), witness.sha256, native_operands)
    return freeze_row_values(projection)
