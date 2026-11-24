"""Capture-time native producer evidence; never a placement or signoff verdict."""
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
from numbers import Real
import time
from uuid import uuid4

from clay.flow.run_workspace import _owned_path, atomic_json


def _identity(path, checkpoint):
    checkpoint()
    digest = sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checkpoint()
            digest.update(block)
            size += len(block)
    checkpoint()
    return {'sha256': digest.hexdigest(), 'size_bytes': size}


def _read_checked_json(path, expected_sha256, checkpoint, *, label, expected_size=None, max_size=None):
    """Hash and parse the very same bytes, with cooperative read checkpoints."""
    checkpoint()
    digest = sha256()
    blocks = []
    size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checkpoint()
            digest.update(block)
            blocks.append(block)
            size += len(block)
            if expected_size is not None and size > expected_size:
                raise ValueError('primitive observation ' + label + ' changed')
            if max_size is not None and size > max_size:
                raise ValueError('primitive observation ' + label + ' exceeds bound')
    checkpoint()
    if digest.hexdigest() != expected_sha256 or (expected_size is not None and size != expected_size):
        raise ValueError('primitive observation ' + label + ' changed')
    return json.loads(b''.join(blocks))


def _read_checked_bytes(path, identity, checkpoint):
    """Return precisely the bytes hashed against a captured output identity."""
    if (not isinstance(identity, dict) or type(identity.get('size_bytes')) is not int
            or identity['size_bytes'] < 0 or type(identity.get('sha256')) is not str):
        raise ValueError('primitive MOS site observation lacks bound output identity')
    digest, blocks, size = sha256(), [], 0
    checkpoint()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checkpoint()
            size += len(block)
            if size > identity['size_bytes']:
                raise ValueError('primitive MOS site observation output changed')
            digest.update(block)
            blocks.append(block)
    checkpoint()
    if size != identity['size_bytes'] or digest.hexdigest() != identity['sha256']:
        raise ValueError('primitive MOS site observation output changed')
    return b''.join(blocks)


def source_snapshot(path, checkpoint):
    """Read the exact source bytes once while honoring the active budget."""
    checkpoint()
    blocks = []
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checkpoint()
            blocks.append(block)
            digest.update(block)
    checkpoint()
    return b''.join(blocks).decode('utf-8').replace('\r\n', '\n').replace('\r', '\n'), digest.hexdigest()


def capture_if_supported(admission, *, provider, **kwargs):
    """Observe the existing native bundle contract; other producers remain UNKNOWN."""
    from clay.layout.native_sky130 import NativeGeometryBundle
    kwargs['checkpoint']()
    if type(kwargs['bundle']) is not NativeGeometryBundle:
        admission.record_primitive_observation_unknown(kwargs['work_dir'], provider.name,
                                                       checkpoint=kwargs['checkpoint'])
        return
    capture(admission, **kwargs)


def _files(root, work, checkpoint):
    root, work = Path(root).absolute(), Path(work).resolve()
    relative = root.relative_to(work)
    _owned_path(work, str(relative))
    if not root.is_dir():
        raise ValueError('primitive producer directory missing')
    output = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        checkpoint()
        base = Path(base)
        for name in sorted(dirs + files):
            path = base / name
            _owned_path(work, str(path.relative_to(work)))
            if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                raise ValueError('primitive producer path alias')
        for name in sorted(files):
            path = base / name
            if not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError('primitive producer file alias')
            output[str(path)] = _identity(path, checkpoint)
    return output


def capture(admission, *, work_dir, geometry_dir, row_dir, bundle, row, cell,
            source_paths, expected_sources, checkpoint):
    """Record one fresh producer result; caller owns the producer invocation.

    The receipt only commits this snapshot. Consumers must replay and recheck
    all paths and bytes before use; cancellation never promotes this to PASS.
    """
    from clay.layout.native_sky130 import NativeGeometryBundle, native_source_identity
    from clay.layout.native_composite_row import NativeCompositeRow

    checkpoint()
    work = Path(work_dir).resolve()
    geometry_dir = Path(geometry_dir).absolute()
    if (type(bundle) is not NativeGeometryBundle or bundle.output_dir != geometry_dir
            or bundle.source_identity != native_source_identity(cell)):
        raise ValueError('primitive producer invocation identity differs')
    _owned_path(work, str(geometry_dir.relative_to(work)))
    if row_dir is None:
        if row is not None:
            raise ValueError('unexpected native composite row')
    elif type(row) is not NativeCompositeRow or row.native_bundle is not bundle:
        raise ValueError('native composite producer invocation identity differs')
    else:
        row_root = Path(row_dir).absolute()
        _owned_path(work, str(row_root.relative_to(work)))
        receipt = Path(row.receipt_path).absolute()
        _owned_path(work, str(receipt.relative_to(work)))
        receipt.relative_to(row_root)
        for _group, _geometry, folder in row.projection.native_witness.rows:
            folder = Path(folder).absolute()
            _owned_path(work, str(folder.relative_to(work)))
            folder.relative_to(row_root)
    # These are the producer's existing live-object and source replay contracts.
    # Their internals do not expose checkpoints; check immediately around each.
    checkpoint()
    bundle.require_current(cell)
    checkpoint()
    if row is not None:
        row.require_current(cell=cell, bundle=bundle)
        checkpoint()
    outputs = _files(geometry_dir, work, checkpoint)
    if row_dir is not None:
        outputs.update(_files(row_dir, work, checkpoint))
    if not outputs:
        raise ValueError('primitive producer emitted no files')
    expected = {str(bundle.output_dir / name): digest for name, digest in bundle.common_artifacts.items()}
    for name, geom in bundle.geometries.items():
        expected.update({str(bundle.folders[name] / path): digest
                         for path, digest in geom.native_evidence['artifacts'].items()})
    dependencies = dict(bundle.dependencies)
    dependencies[str(Path(inspect.getsourcefile(type(bundle))).resolve())] = bundle.producer_sha256
    if row is not None:
        for path, digest in row.dependencies:
            if path in dependencies and dependencies[path] != digest:
                raise ValueError('primitive producer dependency identities disagree')
            dependencies[path] = digest
    for path, digest in expected.items():
        if path not in outputs or outputs[path]['sha256'] != digest:
            raise ValueError('primitive producer output changed during capture')
    dependency_identities = {}
    for path, digest in sorted(dependencies.items()):
        identity = outputs.get(str(path)) or _identity(path, checkpoint)
        if identity['sha256'] != digest:
            raise ValueError('primitive producer dependency changed during capture')
        dependency_identities[str(path)] = identity
    sources = {str(Path(path).resolve()): _identity(path, checkpoint) for path in source_paths}
    if any(sources.get(str(Path(path).resolve()), {}).get('sha256') != digest
           for path, digest in expected_sources.items()):
        raise ValueError('primitive source changed during production')
    members = sorted(bundle.geometries)
    consumed = [] if row is None else sorted(row.program.consumed_devices)
    if not set(consumed) <= set(members):
        raise ValueError('primitive row partition differs from source')
    payload = {
        'schema': 'clay.primitive-producer-observation/v1',
        'scope': 'capture-time-only; fresh replay required; no stage or physical admission',
        'invocation_id': admission.invocation_id,
        'source_identity': bundle.source_identity,
        'source_files': sources,
        'producer_sha256': bundle.producer_sha256,
        'members': members, 'row_members': consumed,
        'standalone_members': sorted(set(members) - set(consumed)),
        'rows': [] if row is None else [
            {'owner': item.owner, 'consumed_devices': list(item.consumed_devices),
             'geometry_sha256': item.geometry_sha256,
             'occurrences': [{'id': occurrence.occurrence_id, 'owner': occurrence.owner}
                             for occurrence in item.geometry.joint_row_binding.occurrences]}
            for item in row.materialization.realizations],
        'row_program_sha256': None if row is None else sha256(row.program.to_canonical_bytes()).hexdigest(),
        'output_roots': [str(geometry_dir)] + ([] if row_dir is None else [str(Path(row_dir).absolute())]),
        'outputs': outputs, 'dependencies': dependency_identities,
    }
    request = getattr(admission, 'role_trace_request', None)
    if request is not None:
        capability_path = geometry_dir / 'role-trace.json'
        capability_identity = outputs.get(str(capability_path))
        if not isinstance(capability_identity, dict):
            raise ValueError('requested role trace manifest is missing from this capture')
        payload['role_trace'] = {'request': request,
                                 'manifest': {'path': str(capability_path), **capability_identity}}
    manifest = admission.attempt_dir / ('primitive-observation-' + uuid4().hex + '.json')
    atomic_json(manifest, payload)
    identity = _identity(manifest, checkpoint)
    checkpoint()
    admission.record_primitive_observation(work, manifest, identity, checkpoint=checkpoint)


def _read_capture_records(*, receipt_path, trusted_receipt_sha256, checkpoint, work_dir=None,
                          _bound_head=None, _require_legacy=False, _receipt_schema='clay.task-attempt/v1'):
    """Read all linked records, optionally filtering to one exact work directory.

    ``trusted_receipt_sha256`` must come from an already trusted RunAdmission
    terminal binding, not a hash computed from arbitrary consumer-supplied JSON.
    Each bounded link is checked back to the first before any record is returned.
    """
    receipt_path = Path(receipt_path).absolute()
    receipt = _read_checked_json(receipt_path, trusted_receipt_sha256, checkpoint,
                                 label='receipt', max_size=64 * 1024)
    allowed_schemas = ((_receipt_schema,) if isinstance(_receipt_schema, str) else _receipt_schema)
    if (receipt_path.name != 'receipt.json' or receipt_path.parent.parent.name != 'attempts'
            or receipt_path.parent.name != receipt.get('invocation_id')
            or receipt.get('schema') not in allowed_schemas):
        raise ValueError('primitive observation receipt path differs from invocation')
    if (receipt.get('schema') in {'clay.primitive-stage-receipt/v1', 'clay.primitive-stage-receipt/v2'}
            and (receipt.get('status') != 'complete' or receipt.get('verdict') != 'UNKNOWN'
                 or receipt.get('physical_admission') is not False)):
        raise ValueError('primitive stage receipt is not a completed observation')
    records = []
    head = receipt.get('primitive_observation_index')
    if _require_legacy and 'primitive_observation_index' in receipt:
        raise ValueError('mixed managed and admission observation formats')
    if ('primitive_observation_index' in receipt
            and ('primitive_observations' in receipt or 'primitive_observation_unknown' in receipt)):
        raise ValueError('mixed primitive observation index and legacy records')
    if 'primitive_observation_index' in receipt and head is None:
        raise ValueError('invalid primitive observation index')
    if _bound_head is not None and (not isinstance(_bound_head, dict)
            or set(_bound_head) != {'path', 'sha256', 'size_bytes', 'count'}
            or type(_bound_head['count']) is not int or _bound_head['count'] < 1):
        raise ValueError('invalid managed primitive observation head')
    if head is not None:
        count = head.get('count') if isinstance(head, dict) else None
        if type(count) is not int or count < 1:
            raise ValueError('invalid primitive observation index count')
        visited = set()
        found_bound_head = _bound_head is None
        while head is not None:
            checkpoint()
            if (not isinstance(head, dict) or set(head) != {'path', 'sha256', 'size_bytes', 'count'}
                    or type(head['count']) is not int or head['count'] != count
                    or type(head['size_bytes']) is not int
                    or not 0 < head['size_bytes'] <= 64 * 1024):
                raise ValueError('invalid primitive observation index link')
            if head == _bound_head:
                found_bound_head = True
            path = Path(head['path']).absolute()
            if path.parent != receipt_path.parent or path in visited:
                raise ValueError('primitive observation index path differs from attempt')
            _owned_path(receipt_path.parent, path.name)
            if path.stat().st_nlink != 1:
                raise ValueError('primitive observation index link alias')
            visited.add(path)
            link = _read_checked_json(path, head['sha256'], checkpoint, label='index link',
                                      expected_size=head['size_bytes'])
            if (link.get('schema') != 'clay.primitive-observation-link/v1'
                    or link.get('invocation_id') != receipt['invocation_id']
                    or link.get('sequence') != count or link.get('kind') not in ('capture', 'unknown')
                    or not isinstance(link.get('record'), dict)):
                raise ValueError('invalid primitive observation index contents')
            if _bound_head is None or count <= _bound_head['count']:
                records.append({'kind': link['kind'], 'record': link['record']})
            head = link.get('previous')
            count -= 1
        if count != 0:
            raise ValueError('primitive observation index is incomplete')
        if not found_bound_head:
            raise ValueError('managed primitive observation head is not bound by receipt')
        records.reverse()
    else:
        if _bound_head is not None:
            raise ValueError('managed primitive observation head is not bound by receipt')
        # Previously issued small receipts remain readable without rewriting them.
        if any(not isinstance(receipt.get(key, []), list)
               for key in ('primitive_observations', 'primitive_observation_unknown')):
            raise ValueError('invalid legacy primitive observations')
        records.extend({'kind': 'capture', 'record': row} for row in receipt.get('primitive_observations', []))
        records.extend({'kind': 'unknown', 'record': row} for row in receipt.get('primitive_observation_unknown', []))
        if any(not isinstance(entry['record'], dict) for entry in records):
            raise ValueError('invalid legacy primitive observation record')
    if work_dir is not None:
        work = str(Path(work_dir).resolve())
        records = [entry for entry in records if entry['record'].get('work_dir') == work]
    return records


def read_capture_records(*, receipt_path, trusted_receipt_sha256, checkpoint, work_dir=None,
                         _bound_head=None, _require_legacy=False):
    return _read_capture_records(receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
        checkpoint=checkpoint, work_dir=work_dir, _bound_head=_bound_head, _require_legacy=_require_legacy)


def read_managed_capture_records(*, terminal_path, trusted_terminal_sha256,
                                 receipt_path, trusted_receipt_sha256, checkpoint):
    """Read a trusted managed terminal's verified prefix for its own work directory."""
    terminal_path = Path(terminal_path).absolute()
    terminal = _read_checked_json(terminal_path, trusted_terminal_sha256, checkpoint, label='terminal')
    index = terminal.get('primitive_observation_index')
    if (terminal_path.name != 'terminal.json' or terminal_path.parent.name != terminal.get('invocation_id')
            or terminal.get('schema') != 'clay.run-terminal/v1'):
        raise ValueError('invalid managed primitive observation scope')
    work_dir = str((terminal_path.parent / 'work').resolve())
    legacy_keys = ('primitive_observations', 'primitive_observation_unknown')
    if 'primitive_observation_index' in terminal:
        if any(key in terminal for key in legacy_keys):
            raise ValueError('mixed managed primitive observation formats')
        if (not isinstance(index, dict) or set(index) != {'head', 'work_dir', 'scope'}
                or index['scope'] != 'only records for this work_dir through this head'
                or index['work_dir'] != work_dir or not isinstance(index['head'], dict)):
            raise ValueError('invalid managed primitive observation scope')
        return read_capture_records(receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
                                    checkpoint=checkpoint, work_dir=work_dir, _bound_head=index['head'])
    if any(not isinstance(terminal.get(key, []), list) for key in legacy_keys):
        raise ValueError('invalid legacy managed primitive observations')
    local = ([{'kind': 'capture', 'record': row} for row in terminal.get(legacy_keys[0], [])]
             + [{'kind': 'unknown', 'record': row} for row in terminal.get(legacy_keys[1], [])])
    parent = read_capture_records(receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
                                  checkpoint=checkpoint, work_dir=work_dir, _require_legacy=True)
    if any(not isinstance(entry['record'], dict) or entry['record'].get('work_dir') != work_dir
           or entry not in parent for entry in local):
        raise ValueError('legacy managed observation is not bound by receipt')
    return local


def _verify_capture_bytes(record, *, receipt_path, trusted_receipt_sha256, checkpoint,
                          _receipt_schema='clay.task-attempt/v1'):
    """Compare retained bytes only; this does not replay the producer or certify a stage.

    ``trusted_receipt_sha256`` must come from an already trusted RunAdmission
    terminal binding, not a hash computed from arbitrary consumer-supplied JSON.
    """
    receipt_path = Path(receipt_path).absolute()
    records = _read_capture_records(receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
                                    checkpoint=checkpoint, _receipt_schema=_receipt_schema)
    if {'kind': 'capture', 'record': record} not in records:
        raise ValueError('primitive observation not bound by receipt')
    manifest = Path(record['path']).absolute()
    if manifest.parent != receipt_path.parent:
        raise ValueError('primitive observation belongs to another attempt')
    _owned_path(receipt_path.parent, manifest.name)
    if manifest.stat().st_nlink != 1:
        raise ValueError('primitive observation manifest alias')
    payload = _read_checked_json(manifest, record['sha256'], checkpoint,
                                 label='manifest', expected_size=record['size_bytes'])
    if payload.get('schema') != 'clay.primitive-producer-observation/v1':
        raise ValueError('invalid primitive observation schema')
    if payload.get('invocation_id') != receipt_path.parent.name:
        raise ValueError('primitive observation invocation differs')
    outputs = {}
    if not Path(record['work_dir']).is_absolute():
        raise ValueError('primitive observation work directory is not absolute')
    for root in payload['output_roots']:
        outputs.update(_files(root, record['work_dir'], checkpoint))
    if outputs != payload['outputs']:
        raise ValueError('primitive producer output changed')
    for paths in (payload['source_files'], payload['dependencies']):
        for path, identity in paths.items():
            if _identity(path, checkpoint) != identity:
                raise ValueError('primitive observation source or dependency changed')
    return payload


def verify_capture_bytes(record, *, receipt_path, trusted_receipt_sha256, checkpoint):
    return _verify_capture_bytes(record, receipt_path=receipt_path,
        trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=checkpoint)


def read_stage_capture_records(*, receipt_path, trusted_receipt_sha256, checkpoint, work_dir=None,
                               managed_terminal_path=None, trusted_terminal_sha256=None):
    """Read a trusted, completed primitive-only stage without granting admission."""
    receipt_path = Path(receipt_path).absolute()
    receipt = _read_checked_json(receipt_path, trusted_receipt_sha256, checkpoint,
                                 label='stage receipt', max_size=64 * 1024)
    start = receipt.get('start')
    if not isinstance(start, dict):
        raise ValueError('primitive stage start identity missing')
    start_path = Path(start['path']).absolute()
    if start_path.parent != receipt_path.parent:
        raise ValueError('primitive stage start belongs to another attempt')
    start_data = _read_checked_json(start_path, start['sha256'], checkpoint,
        label='stage start', expected_size=start['size_bytes'])
    pair = (start_data.get('schema'), receipt.get('schema'))
    if pair not in {('clay.primitive-stage-start/v1', 'clay.primitive-stage-receipt/v1'),
                    ('clay.primitive-stage-start/v2', 'clay.primitive-stage-receipt/v2')}:
        raise ValueError('primitive stage schema pair differs')
    if pair[0] == 'clay.primitive-stage-start/v2':
        from clay.flow.run_admission import _checked_role_trace_request
        request = _checked_role_trace_request(start_data.get('role_trace_request'), start_data.get('budget'))
        if receipt.get('role_trace_request') != request:
            raise ValueError('primitive stage role trace request differs')
    elif 'role_trace_request' in start_data or 'role_trace_request' in receipt:
        raise ValueError('legacy primitive stage cannot bind a role trace request')
    if (type(receipt.get('managed')) is not bool or start_data.get('managed') != receipt['managed']
            or start_data.get('task_id') != receipt.get('task_id')
            or start_data.get('invocation_id') != receipt.get('invocation_id')):
        raise ValueError('primitive stage start scope differs')
    bindings_id = start_data.get('source_bindings')
    if not isinstance(bindings_id, dict):
        raise ValueError('primitive stage source bindings missing')
    bindings_path = Path(bindings_id['path']).absolute()
    if bindings_path.parent != receipt_path.parent:
        raise ValueError('primitive stage source bindings belong to another attempt')
    bindings = _read_checked_json(bindings_path, bindings_id['sha256'], checkpoint,
        label='stage source bindings', expected_size=bindings_id['size_bytes'], max_size=16 * 1024 * 1024)
    if not isinstance(bindings, dict) or set(bindings) != {'design', 'spice', 'runtime', 'dependencies'}:
        raise ValueError('primitive stage source bindings malformed')
    entries = [bindings[key] for key in ('design', 'spice', 'runtime') if bindings[key] is not None]
    if not isinstance(bindings['dependencies'], list) or not bindings['dependencies']:
        raise ValueError('primitive stage dependency bindings missing')
    entries.extend(bindings['dependencies'])
    for identity in entries:
        if (not isinstance(identity, dict) or set(identity) != {'path', 'sha256', 'size_bytes'}
                or not Path(identity['path']).is_absolute()
                or _identity(identity['path'], checkpoint) !=
                   {'sha256': identity['sha256'], 'size_bytes': identity['size_bytes']}):
            raise ValueError('primitive stage source or dependency changed')
    bound_head = None
    if receipt['managed']:
        if managed_terminal_path is None or trusted_terminal_sha256 is None:
            raise ValueError('managed primitive stage requires a trusted terminal')
        terminal_path = Path(managed_terminal_path).absolute()
        terminal_id = receipt.get('managed_terminal')
        if (not isinstance(terminal_id, dict) or str(terminal_path) != terminal_id.get('path')
                or trusted_terminal_sha256 != terminal_id.get('sha256')):
            raise ValueError('managed primitive terminal identity differs')
        terminal = _read_checked_json(terminal_path, trusted_terminal_sha256, checkpoint,
            label='managed stage terminal', expected_size=terminal_id['size_bytes'])
        index = terminal.get('primitive_observation_index')
        if (terminal.get('schema') != 'clay.run-terminal/v1' or terminal.get('status') != 'complete'
                or terminal.get('verdict') != 'unknown' or terminal.get('invocation_id') != terminal_path.parent.name
                or not isinstance(index, dict) or index.get('head') != receipt.get('primitive_observation_index')
                or index.get('work_dir') != str((terminal_path.parent/'work').resolve())
                or index.get('scope') != 'only records for this work_dir through this head'):
            raise ValueError('managed primitive terminal scope differs')
        bound_head = index['head']
        work_dir = index['work_dir'] if work_dir is None else work_dir
        if work_dir != index['work_dir']:
            raise ValueError('managed primitive work directory differs')
    elif managed_terminal_path is not None or trusted_terminal_sha256 is not None or receipt.get('managed_terminal') is not None:
        raise ValueError('legacy primitive stage cannot have a managed terminal')
    records = _read_capture_records(receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
        checkpoint=checkpoint, work_dir=work_dir, _bound_head=bound_head,
        _receipt_schema=('clay.primitive-stage-receipt/v1', 'clay.primitive-stage-receipt/v2'))
    if pair[0] == 'clay.primitive-stage-start/v2':
        plan_id = receipt.get('resource_plan')
        capability = receipt.get('role_trace_manifest')
        if (not isinstance(plan_id, dict) or Path(plan_id.get('path', '')).parent != receipt_path.parent
                or not isinstance(capability, dict) or len(records) != 1
                or records[0].get('kind') != 'capture'):
            raise ValueError('primitive role trace receipt binding missing')
        plan = _read_checked_json(plan_id['path'], plan_id['sha256'], checkpoint,
                                  label='container plan', expected_size=plan_id['size_bytes'])
        if (plan.get('schema') != 'clay.native-container-plan/v2'
                or plan.get('role_trace_request') != request
                or Path(capability.get('path', '')) != Path(plan['work_dir']) / 'role-trace.json'):
            raise ValueError('primitive role trace plan binding differs')
        record = records[0]['record']
        payload = _read_checked_json(record['path'], record['sha256'], checkpoint,
                                    label='capture', expected_size=record['size_bytes'])
        expected = {'request': request, 'manifest': capability}
        if (payload.get('role_trace') != expected
                or payload.get('outputs', {}).get(capability['path']) !=
                   {'sha256': capability['sha256'], 'size_bytes': capability['size_bytes']}
                or _identity(capability['path'], checkpoint) !=
                   {'sha256': capability['sha256'], 'size_bytes': capability['size_bytes']}):
            raise ValueError('primitive role trace capture binding differs')
    return records


def verify_stage_capture_bytes(record, *, receipt_path, trusted_receipt_sha256, checkpoint,
                               managed_terminal_path=None, trusted_terminal_sha256=None):
    """Verify capture bytes; this does not revalidate resource supervision."""
    records = read_stage_capture_records(receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
        checkpoint=checkpoint, managed_terminal_path=managed_terminal_path,
        trusted_terminal_sha256=trusted_terminal_sha256)
    if {'kind': 'capture', 'record': record} not in records:
        raise ValueError('primitive stage capture not bound')
    return _verify_capture_bytes(record, receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
        checkpoint=checkpoint, _receipt_schema=('clay.primitive-stage-receipt/v1', 'clay.primitive-stage-receipt/v2'))


def _validated_unwind_budget(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('primitive unwind budget must be finite and positive')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError('primitive unwind budget must be finite and positive') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError('primitive unwind budget must be finite and positive')
    return value


def replay_standalone_capture(record, *, receipt_path, trusted_receipt_sha256,
                              source_path, checkpoint, terminal_path=None,
                              trusted_terminal_sha256=None, unwind_budget_seconds=None,
                              _mos_site_details=None, _stage_receipt=False):
    """Replay captured native masters against their source, without stage admission.

    Receipt (and, when given, managed terminal) digests must come from a
    separately trusted terminal binding. Composite rows are not replayed by
    this operation and remain UNKNOWN. Reopen and recheck all captured bytes
    around semantic observation; never reconstruct a live producer authority.
    An optional finite unwind budget enables a separate cooperative byte
    recheck after cancellation. Individual filesystem calls are not preempted.
    """
    from clay.flow.control_stop import record_unwind_failure
    from clay.flow.execution_budget import ExecutionStopped
    from clay.layout.native_sky130 import (
        _cell_native_specs, _digest, _geometry_identity, native_source_identity,
        observe_native_primitive,
    )
    from clay.parser import parse_clay_source

    if (terminal_path is None) != (trusted_terminal_sha256 is None):
        raise ValueError('managed terminal path and trusted digest are both required')
    unwind_budget_seconds = _validated_unwind_budget(unwind_budget_seconds)

    def verify(check=checkpoint):
        if _stage_receipt:
            return verify_stage_capture_bytes(record, receipt_path=receipt_path,
                trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=check,
                managed_terminal_path=terminal_path, trusted_terminal_sha256=trusted_terminal_sha256)
        if terminal_path is not None:
            entries = read_managed_capture_records(
                terminal_path=terminal_path, trusted_terminal_sha256=trusted_terminal_sha256,
                receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
                checkpoint=check)
            if {'kind': 'capture', 'record': record} not in entries:
                raise ValueError('primitive observation not bound by managed terminal')
        return verify_capture_bytes(record, receipt_path=receipt_path,
                                    trusted_receipt_sha256=trusted_receipt_sha256,
                                    checkpoint=check)

    payload = None
    try:
        payload = verify()
        source = Path(source_path).resolve(strict=True)
        identity = payload.get('source_files', {}).get(str(source))
        if source.suffix.casefold() != '.clay' or not isinstance(identity, dict):
            raise ValueError('primitive observation source is not bound')
        source_bytes = source.read_bytes()
        checkpoint()
        if sha256(source_bytes).hexdigest() != identity.get('sha256') or len(source_bytes) != identity.get('size_bytes'):
            raise ValueError('primitive observation source changed')
        parsed = parse_clay_source(source_bytes.decode('utf-8-sig'))
        cells = [cell for cell in parsed.cell_defs if native_source_identity(cell) == payload.get('source_identity')]
        if len(cells) != 1:
            raise ValueError('primitive observation source identity differs')
        cell, = cells
        from clay.layout.intent import layout_intent_from_cell_def
        intent = layout_intent_from_cell_def(cell)
        if (payload.get('row_members') or payload.get('rows') or payload.get('row_program_sha256') is not None
                or any(row.style == 'native_diffusion' for row in intent.arrays)):
            result = {'schema': 'clay.primitive-semantic-observation/v1',
                      'status': 'UNKNOWN', 'scope': 'composite row semantic replay unavailable',
                      'physical_admission': False,
                      'unknown': ['COMPOSITE_ROW_REPLAY', 'BODY_SEMICONDUCTOR_COVERAGE',
                                  'CUT_GEOMETRY_COMPLETENESS', 'REFERENCE_OCCURRENCE_CORRESPONDENCE',
                                  'PRIMITIVE_STAGE_ADMISSION']}
        else:
            specs = _cell_native_specs(cell)
            names = sorted(specs)
            if (not names or payload.get('members') != names or payload.get('standalone_members') != names
                    or payload.get('row_members') != [] or payload.get('rows') != []):
                raise ValueError('primitive observation source member denominator differs')
            roots = payload.get('output_roots')
            if not isinstance(roots, list) or len(roots) != 1:
                raise ValueError('primitive observation standalone output root differs')
            root = Path(roots[0])
            geometry_path = root / 'geometry.json'
            geometry_identity = payload.get('outputs', {}).get(str(geometry_path))
            if not isinstance(geometry_identity, dict):
                raise ValueError('primitive observation geometry manifest is not bound')
            geometry = _read_checked_json(geometry_path, geometry_identity['sha256'], checkpoint,
                                          label='geometry manifest', expected_size=geometry_identity['size_bytes'])
            if (geometry.get('source_identity') != payload['source_identity']
                    or geometry.get('producer_sha256') != payload.get('producer_sha256')
                    or set(geometry.get('members', {})) != set(names)
                    or set(geometry.get('geometry_identities', {})) != set(names)):
                raise ValueError('primitive observation geometry source or member denominator differs')
            observed = {}
            site_observations = {}
            for name in names:
                checkpoint()
                spec = specs[name]
                key = 'pcell_' + _digest(spec)[:20]
                if (geometry['members'][name] != key
                        or geometry.get('primitives', {}).get(key, {}).get('source_spec') != spec):
                    raise ValueError('primitive observation source specification differs: ' + name)
                if key not in observed:
                    if _mos_site_details is not None and spec['family'] == 'mos':
                        site_sink = []
                        folder = root / key
                        names_to_read = ('drc.txt', 'primitive.gds', 'native.spice',
                                         'scale_um.txt', 'primitive_flat.ext')
                        bound_inputs = {item: _read_checked_bytes(folder / item,
                            payload['outputs'].get(str(folder / item)), checkpoint)
                            for item in names_to_read}
                        native_geometry = observe_native_primitive(root / key, spec, _mos_site_sink=site_sink,
                            _mos_site_checkpoint=checkpoint, _bound_inputs=bound_inputs)
                        if len(site_sink) != 1:
                            raise ValueError('primitive MOS site observation is incomplete')
                        site_observations[key] = site_sink[0]
                    else:
                        native_geometry = observe_native_primitive(root / key, spec)
                    observed[key] = _geometry_identity(native_geometry)
                    checkpoint()
                if observed[key] != geometry['geometry_identities'][name]:
                    raise ValueError('primitive observation geometry identity differs: ' + name)
            if set(geometry.get('primitives', {})) != set(observed):
                raise ValueError('primitive observation master denominator differs')
            result = {'schema': 'clay.primitive-semantic-observation/v1',
                      'status': 'UNKNOWN', 'scope': 'standalone source and native master replay only',
                      'physical_admission': False, 'source_identity': payload['source_identity'],
                      'source_sha256': identity['sha256'], 'member_count': len(names),
                      'master_count': len(observed), 'source_spec_geometry': 'PASS',
                      'unknown': ['BODY_SEMICONDUCTOR_COVERAGE', 'CUT_GEOMETRY_COMPLETENESS',
                                  'REFERENCE_OCCURRENCE_CORRESPONDENCE', 'PRIMITIVE_STAGE_ADMISSION']}
        if verify() != payload:
            raise ValueError('primitive observation changed during semantic replay')
        if _mos_site_details is not None and result.get('source_spec_geometry') == 'PASS':
            by_name = {device.instance_name: device for device in cell.devices}
            _mos_site_details.extend(
                (name, tuple(sorted(by_name[name].pins.items())),
                 site_observations['pcell_' + _digest(specs[name])[:20]])
                for name in names if specs[name]['family'] == 'mos')
    except BaseException as original:
        if isinstance(original, ExecutionStopped):
            status = 'incomplete'
            if unwind_budget_seconds is None:
                ending = TimeoutError('PRIMITIVE_UNWIND_VERIFICATION_INCOMPLETE: no unwind budget')
                record_unwind_failure(original, ending)
            else:
                deadline = time.monotonic() + unwind_budget_seconds
                def unwind_checkpoint():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('PRIMITIVE_UNWIND_VERIFICATION_INCOMPLETE: unwind budget exhausted')
                try:
                    checked = verify(unwind_checkpoint)
                    if payload is not None and checked != payload:
                        raise ValueError('primitive observation changed during semantic replay')
                    status = 'complete'
                except BaseException as ending:
                    status = 'incomplete' if isinstance(ending, TimeoutError) else 'failed'
                    record_unwind_failure(original, ending)
            try:
                BaseException.add_note(original, f'Primitive unwind verification: budget_seconds={unwind_budget_seconds}; status={status}')
            except BaseException:
                pass
            raise
        try:
            verify()
        except BaseException as ending:
            record_unwind_failure(original, ending)
        raise
    return result


def _mos_site_dependencies():
    """Files used by the optional observation; not loaded-code authority."""
    from clay.layout import (connectivity, connectivity_queries, native_sky130,
                             occurrence_channels, physical_terminal_inventory,
                             terminal_body_geometry)
    from clay.verification import native_rc
    modules = (connectivity, connectivity_queries, native_sky130, occurrence_channels,
               physical_terminal_inventory, terminal_body_geometry, native_rc)
    return tuple(sorted({str(Path(__file__).resolve()),
                         *(str(Path(module.__file__).resolve()) for module in modules)}))


def replay_standalone_mos_sites(record, *, receipt_path, trusted_receipt_sha256,
                                source_path, checkpoint, terminal_path=None,
                                trusted_terminal_sha256=None, unwind_budget_seconds=None,
                                _stage_receipt=False):
    """Reopen native MOS channel sites under the existing capture contract.

    Source terminal nets are retained as a separate unordered-end contract;
    this observation does not assign a diffusion side to a source terminal.
    """
    from clay.flow.control_stop import record_unwind_failure
    from clay.flow.execution_budget import ExecutionStopped

    unwind_budget_seconds = _validated_unwind_budget(unwind_budget_seconds)
    dependencies = None
    phase = 'consumer-before'
    try:
        dependencies = {path: _identity(path, checkpoint) for path in _mos_site_dependencies()}
        details = []
        phase = 'semantic'
        semantic = replay_standalone_capture(
            record, receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
            source_path=source_path, checkpoint=checkpoint, terminal_path=terminal_path,
            trusted_terminal_sha256=trusted_terminal_sha256,
            unwind_budget_seconds=unwind_budget_seconds, _mos_site_details=details,
            _stage_receipt=_stage_receipt)
        phase = 'consumer-after'
        if {path: _identity(path, checkpoint) for path in dependencies} != dependencies:
            raise ValueError('primitive MOS site observation consumer dependency changed')
    except ExecutionStopped as original:
        if phase == 'semantic':
            raise
        status = 'incomplete'
        if unwind_budget_seconds is None or dependencies is None:
            ending = TimeoutError('PRIMITIVE_UNWIND_VERIFICATION_INCOMPLETE: no complete consumer snapshot or unwind budget')
            record_unwind_failure(original, ending)
        else:
            deadline = time.monotonic() + unwind_budget_seconds
            def unwind_checkpoint():
                if time.monotonic() >= deadline:
                    raise TimeoutError('PRIMITIVE_UNWIND_VERIFICATION_INCOMPLETE: unwind budget exhausted')
            try:
                if _stage_receipt:
                    verify_stage_capture_bytes(record, receipt_path=receipt_path,
                        trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=unwind_checkpoint,
                        managed_terminal_path=terminal_path, trusted_terminal_sha256=trusted_terminal_sha256)
                elif terminal_path is not None:
                    entries = read_managed_capture_records(
                        terminal_path=terminal_path, trusted_terminal_sha256=trusted_terminal_sha256,
                        receipt_path=receipt_path, trusted_receipt_sha256=trusted_receipt_sha256,
                        checkpoint=unwind_checkpoint)
                    if {'kind': 'capture', 'record': record} not in entries:
                        raise ValueError('primitive observation not bound by managed terminal')
                if not _stage_receipt:
                    verify_capture_bytes(record, receipt_path=receipt_path,
                        trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=unwind_checkpoint)
                if {path: _identity(path, unwind_checkpoint) for path in dependencies} != dependencies:
                    raise ValueError('primitive MOS site observation consumer dependency changed')
                status = 'complete'
            except BaseException as ending:
                status = 'incomplete' if isinstance(ending, TimeoutError) else 'failed'
                record_unwind_failure(original, ending)
        try:
            BaseException.add_note(original, f'Primitive MOS consumer unwind verification: budget_seconds={unwind_budget_seconds}; status={status}')
        except BaseException:
            pass
        raise
    return {'schema': 'clay.native-mos-channel-site-observation/v1',
            'status': 'UNKNOWN', 'physical_admission': False,
            'source_identity': semantic.get('source_identity'),
            'consumer_file_stability': dependencies,
            'source_sha256': semantic.get('source_sha256'),
            'producer_receipt_sha256': trusted_receipt_sha256,
            'managed_terminal_sha256': trusted_terminal_sha256,
            'members': tuple(details),
            'unknown': tuple(dict.fromkeys((*semantic.get('unknown', ()),
                        'LOADED_CONSUMER_IDENTITY', 'EXT_SPICE_OCCURRENCE_BINDING',
                        'SOURCE_GATE_POLY_NET_BINDING', 'SOURCE_TERMINAL_SIDE_BINDING',
                        'SOURCE_BODY_NET_BINDING', 'BODY_SEMICONDUCTOR_COVERAGE',
                        'CUT_GEOMETRY_COMPLETENESS', 'NON_MOS_SITES',
                        'REFERENCE_OCCURRENCE_CORRESPONDENCE',
                        'PRIMITIVE_STAGE_ADMISSION')))}


def replay_stage_standalone_capture(record, *, receipt_path, trusted_receipt_sha256,
                                    source_path, checkpoint, terminal_path=None,
                                    trusted_terminal_sha256=None, unwind_budget_seconds=None):
    """Replay a stage receipt without treating it as a production receipt."""
    observed = replay_standalone_capture(record, receipt_path=receipt_path,
        trusted_receipt_sha256=trusted_receipt_sha256, source_path=source_path, checkpoint=checkpoint,
        terminal_path=terminal_path, trusted_terminal_sha256=trusted_terminal_sha256,
        unwind_budget_seconds=unwind_budget_seconds, _stage_receipt=True)
    observed['unknown'] = tuple(dict.fromkeys((*observed.get('unknown', ()),
        'RESOURCE_SUPERVISION_FRESHNESS', 'INPUT_CONSUMPTION_IDENTITY', 'LOADED_CODE_IDENTITY',
        'LOADED_DOCKER_CLI_IDENTITY', 'TOOLKIT_SOURCE_EXPECTATION')))
    return observed


def replay_stage_mos_sites(record, *, receipt_path, trusted_receipt_sha256,
                           source_path, checkpoint, terminal_path=None,
                           trusted_terminal_sha256=None, unwind_budget_seconds=None):
    observed = replay_standalone_mos_sites(record, receipt_path=receipt_path,
        trusted_receipt_sha256=trusted_receipt_sha256, source_path=source_path, checkpoint=checkpoint,
        terminal_path=terminal_path, trusted_terminal_sha256=trusted_terminal_sha256,
        unwind_budget_seconds=unwind_budget_seconds, _stage_receipt=True)
    observed['unknown'] = tuple(dict.fromkeys((*observed.get('unknown', ()),
        'RESOURCE_SUPERVISION_FRESHNESS', 'INPUT_CONSUMPTION_IDENTITY', 'LOADED_CODE_IDENTITY',
        'LOADED_DOCKER_CLI_IDENTITY', 'TOOLKIT_SOURCE_EXPECTATION')))
    return observed

