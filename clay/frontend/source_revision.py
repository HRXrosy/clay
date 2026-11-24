"""Replay full electrical identity across physical authoring revisions.

These records describe the files actually reopened here. They do not establish
native geometry, reference ownership, source-projection registry, or physical
admission. Consumers retain their original input bindings and independently
capture/replay the new source at their own native boundary.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

_REOPEN_BY_PATH_SHA: dict[tuple[str, str], tuple[str, tuple[tuple[str, object, dict, str, str], ...]]] = {}
_REOPEN_CACHE_MAX = 32


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def electrical_source_data(cell):
    """Canonical complete CellDef semantics, excluding physical declarations."""
    from clay.ir import CellDef
    from clay.layout.route.exact_artifact_evidence import _canonical_source_member_inventory
    _require(type(cell) is CellDef, 'complete source requires an exact parsed CellDef')

    def semantic(value):
        if is_dataclass(value):
            return {field.name: semantic(getattr(value, field.name)) for field in fields(value)
                    if field.name != 'source_span'}
        if type(value) is dict:
            return {key: semantic(item) for key, item in value.items() if key != 'source_span'}
        if type(value) in (list, tuple):
            return [semantic(item) for item in value]
        return value

    # Enumerate actual fields so future nonphysical fields cannot silently drop.
    # Ports remain ordered; all device models/pins/sizing are canonicalized.
    # Dynamic parser _layout_annotations are physical declaration data.
    result = {field.name: semantic(getattr(cell, field.name)) for field in fields(cell)
              if field.name not in ('devices', 'layout_intents', 'source_span')}
    result['devices'] = _canonical_source_member_inventory(cell)
    _digest(result)
    return result


def reopen_complete_source(cell):
    """Reopen a uniquely selected current cell and check the supplied semantics."""
    from clay.layout.route.exact_artifact_evidence import _source_member_inventory_sha256
    from clay import parser as _clay_parser
    parse_clay_file = _clay_parser.parse_clay_file
    supplied = electrical_source_data(cell)
    _require(type(cell.source_span) is dict and type(cell.source_span.get('file')) is str,
             'complete source lacks actual file provenance')
    path = Path(cell.source_span['file']).resolve(strict=True)
    before = _file_sha(path)
    cache_key = (str(path), before, id(parse_clay_file))
    cached = _REOPEN_BY_PATH_SHA.get(cache_key)
    if cached is not None:
        _require(_file_sha(path) == before, 'current source changed during complete electrical replay')
        pdk_id, entries = cached
        matched_rows = tuple(row for row in entries if row[0] == cell.local_name)
        _require(len(matched_rows) == 1, 'complete source cell identity is not unique')
        _, canonical_tpl, electrical_tpl, electrical_sha, inv_sha = matched_rows[0]
        _require(_digest(supplied) == electrical_sha,
                 'provided source cell electrical inventory differs from current source bytes')
        _require(type(pdk_id) is str and bool(pdk_id), 'complete source PDK declaration is missing')
        return dict(
            cell=deepcopy(canonical_tpl),
            path=str(path),
            source_sha256=before,
            source_inventory_sha256=inv_sha,
            electrical=deepcopy(electrical_tpl),
            electrical_sha256=electrical_sha,
            pdk_id=pdk_id,
        )
    parsed = parse_clay_file(path)
    _require(_file_sha(path) == before, 'current source changed during complete electrical replay')
    matches = tuple(item for item in parsed.cell_defs if item.local_name == cell.local_name)
    _require(len(matches) == 1, 'complete source cell identity is not unique')
    canonical, = matches
    electrical = electrical_source_data(canonical)
    electrical_sha = _digest(electrical)
    _require(_digest(supplied) == electrical_sha,
             'provided source cell electrical inventory differs from current source bytes')
    _require(type(parsed.pdk_id) is str and bool(parsed.pdk_id), 'complete source PDK declaration is missing')
    inv_sha = _source_member_inventory_sha256(canonical)
    if type(parsed.pdk_id) is str and bool(parsed.pdk_id):
        all_rows = []
        for item in parsed.cell_defs:
            item_elec = electrical if item is canonical else electrical_source_data(item)
            item_elec_sha = electrical_sha if item is canonical else _digest(item_elec)
            item_inv_sha = inv_sha if item is canonical else _source_member_inventory_sha256(item)
            all_rows.append((item.local_name, deepcopy(item), deepcopy(item_elec), item_elec_sha, item_inv_sha))
        if len(_REOPEN_BY_PATH_SHA) >= _REOPEN_CACHE_MAX:
            _REOPEN_BY_PATH_SHA.pop(next(iter(_REOPEN_BY_PATH_SHA)))
        _REOPEN_BY_PATH_SHA[cache_key] = (parsed.pdk_id, tuple(all_rows))
    return dict(cell=canonical, path=str(path), source_sha256=before,
        source_inventory_sha256=inv_sha,
        electrical=electrical, electrical_sha256=electrical_sha, pdk_id=parsed.pdk_id)


def replay_source_revision(original_source_cell, source_cell, *, tech):
    """Require full electrical equivalence and the current declared PDK.

The original record's SHA must still match the consumer's independent original
capture. This function never changes that capture or any intrinsic recipe hash.
    """
    from clay.layout.tech import LayoutTech
    original = reopen_complete_source(original_source_cell)
    current = reopen_complete_source(source_cell)
    _require(original['electrical_sha256'] == current['electrical_sha256'],
             'new source electrical/device/port inventory differs from complete original source')
    _require(LayoutTech.resolve(original['pdk_id']).pdk_id
             == LayoutTech.resolve(current['pdk_id']).pdk_id == tech.pdk_id,
             'authoring source PDK identity differs from current construction')
    snapshot = tech.adapter.source_snapshot()
    identity = (tech.pdk_id, None if snapshot is None else snapshot.snapshot_sha256,
                None if snapshot is None else snapshot.installed_authority_sha256)
    for record in (original, current):
        _require(_file_sha(record['path']) == record['source_sha256'],
                 'source bytes changed during complete authoring revision')
    return dict(original=original, current=current, pdk_identity=identity)


__all__ = ['reopen_complete_source', 'replay_source_revision']
