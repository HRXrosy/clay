"""Versioned, read-only recovery of an interrupted pre-create observation.

The retained supervisor evidence is authoritative only for its recorded command.
The new scans observe an endpoint now; they do not identify the historical daemon.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time
import difflib

from clay.flow.native_owned_container import _cli_identity
from clay.flow.owned_process import run_owned_process
from clay.flow.run_workspace import atomic_json

_ENDPOINT = 'npipe:////./pipe/docker_engine'
_SCHEMA = 'clay.primitive-precreate-recovery-evidence/v1'
_SCHEMA_DESCENDANTS = 'clay.primitive-precreate-recovery-evidence/v2'
_DESKTOP_ENDPOINT = 'npipe:////./pipe/dockerDesktopLinuxEngine'
_MAX = 256 * 1024 * 1024


def _read(path, checkpoint):
    chunks = []
    size = 0
    with Path(path).open('rb') as stream:
        while True:
            checkpoint()
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > _MAX:
                raise ValueError('recovery input exceeds byte limit')
            chunks.append(block)
    checkpoint()
    return b''.join(chunks)


def _identity(path, checkpoint):
    path = Path(path).resolve(strict=True)
    data = _read(path, checkpoint)
    return {'path': str(path), 'sha256': sha256(data).hexdigest(), 'size_bytes': len(data)}


def _checked(identity, checkpoint):
    if not isinstance(identity, dict) or set(identity) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('invalid recovery file identity')
    data = _read(identity['path'], checkpoint)
    if (str(Path(identity['path']).resolve(strict=True)) != identity['path']
            or len(data) != identity['size_bytes'] or sha256(data).hexdigest() != identity['sha256']):
        raise ValueError('recovery file identity differs')
    return data


def _archive(item, checkpoint, *, check_original=True):
    archive = {'path': item['archive'], 'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
    data = _checked(archive, checkpoint)
    if check_original:
        original = item.get('original') or {'path': item['original_path'],
            'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
        if _checked(original, checkpoint) != data:
            raise ValueError('historical original differs from archive')
    return data


def _save_snapshot(directory, label, data, checkpoint):
    path = directory / (label + '-' + sha256(data).hexdigest() + '.bin')
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(data)
    identity = _identity(path, checkpoint)
    if identity['sha256'] != sha256(data).hexdigest() or identity['size_bytes'] != len(data):
        raise ValueError('recovery snapshot differs')
    return identity


def _historical(admission, active, preparation_identity, checkpoint, *, evidence=None, snapshot_dir=None):
    manifest = json.loads(_checked(preparation_identity, checkpoint))
    preparation_schema = manifest.get('schema')
    descendant_case = preparation_schema == 'clay.primitive-precreate-recovery-preparation/v2'
    if (preparation_schema not in ('clay.primitive-precreate-recovery-preparation/v1',
                                   'clay.primitive-precreate-recovery-preparation/v2')
            or (descendant_case and manifest.get('failure_kind') != 'owned-ps-descendant-termination')
            or manifest.get('task_id') != admission.task_id
            or manifest.get('invocation_id') != active['invocation_id']
            or manifest.get('stage_start') != active['start']
            or manifest.get('resource_plan') != active['plan']):
        raise ValueError('recovery preparation does not identify active stage')
    old = manifest['evidence']
    # state.json legitimately advances after terminal persistence; use archived state.
    state = json.loads(_archive(old['state'], checkpoint, check_original=False))
    if state['primitive_stage']['active'] != active:
        raise ValueError('archived active stage differs')
    live = evidence is None
    start = json.loads(_archive(old['stage_start'], checkpoint, check_original=live))
    bindings = json.loads(_archive(old['source_bindings'], checkpoint, check_original=live))
    plan = json.loads(_archive(old['resource_plan'], checkpoint, check_original=live))
    supervision = json.loads(_archive(old['supervision'], checkpoint, check_original=live))
    out = _archive(old['baseline_stdout'], checkpoint, check_original=live)
    err = _archive(old['baseline_stderr'], checkpoint, check_original=live)
    if descendant_case:
        events = _archive(old['events'], checkpoint, check_original=live)
        if Path(old['events']['original_path']) != Path(active['start']['path']).parent / 'events.jsonl':
            raise ValueError('historical failure event belongs to another attempt')
        rows = [json.loads(line) for line in events.splitlines() if line.strip()]
        if (not rows or rows[-1].get('event') != 'phase_error'
                or rows[-1].get('phase') != 'flow.primitive-observe'
                or rows[-1].get('invocation_id') != active['invocation_id']
                or rows[-1].get('error') != {'error': 'owned Docker command did not complete: ps',
                                              'error_type': 'ValueError'}):
            raise ValueError('historical typed failure event differs')
        quiescence = start.get('production_quiescence')
        if not isinstance(quiescence, dict) or quiescence.get('endpoint') != _DESKTOP_ENDPOINT:
            raise ValueError('historical stage current endpoint differs')
    def old_identity(item):
        return {'path': item['original_path'], 'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
    work = Path(plan['work_dir'])
    if (old_identity(old['stage_start']) != active['start']
            or old_identity(old['resource_plan']) != active['plan']
            or old_identity(old['source_bindings']) != start['source_bindings']
            or Path(old['supervision']['original_path']) != work / 'native-supervision.json'
            or Path(old['baseline_stdout']['original_path']) != work / 'owned-00.stdout'
            or Path(old['baseline_stderr']['original_path']) != work / 'owned-00.stderr'
            or start.get('managed') is not False
            or start['source_bindings'] != manifest['source_bindings']
            or start['task_id'] != admission.task_id or start['invocation_id'] != active['invocation_id']
            or plan['invocation_id'] != active['invocation_id']
            or plan['name'] != 'clay-primitive-' + active['invocation_id']):
        raise ValueError('historical source or plan differs')
    expected = [bindings[k] for k in ('design', 'spice', 'runtime')] + bindings['dependencies']
    archived = manifest['bindings']
    if len(expected) != len(archived) or [x['original'] for x in archived] != expected:
        raise ValueError('historical archive does not cover source closure')
    changed = []
    retained = {} if live else {row['original']['path']: row for row in evidence['historical_changed_paths']}
    if not live and len(retained) != len(evidence['historical_changed_paths']):
        raise ValueError('duplicate recovery changed dependency')
    clay = Path(__file__).resolve().parent.parent
    allowed = {str((clay / 'flow' / 'run_admission.py').resolve()),
               str((clay / 'api.py').resolve()), str((clay / 'cli.py').resolve())}
    if descendant_case:
        allowed.add(str(Path(__file__).resolve()))
    for item in archived:
        old_bytes = _archive(item, checkpoint, check_original=False)
        original = item['original']
        if sha256(old_bytes).hexdigest() != original['sha256'] or len(old_bytes) != original['size_bytes']:
            raise ValueError('historical archive differs from source binding')
        retained_row = retained.get(original['path'])
        new_bytes = (_read(original['path'], checkpoint) if live else
                     _checked(retained_row['snapshot'], checkpoint) if retained_row else old_bytes)
        new = {'path': original['path'], 'sha256': sha256(new_bytes).hexdigest(), 'size_bytes': len(new_bytes)}
        if new != original:
            if original['path'] not in allowed or original not in bindings['dependencies']:
                raise ValueError('unrelated historical dependency changed')
            diff = b''.join(difflib.diff_bytes(difflib.unified_diff, old_bytes.splitlines(keepends=True),
                new_bytes.splitlines(keepends=True), fromfile=b'historical', tofile=b'current'))
            snapshot = (_save_snapshot(snapshot_dir, 'changed', new_bytes, checkpoint)
                        if live else retained_row['snapshot'])
            changed.append({'original': original, 'current': new, 'snapshot': snapshot,
                            'diff_sha256': sha256(diff).hexdigest(), 'diff_size_bytes': len(diff),
                            'diff_utf8': diff.decode('utf-8')})
    inputs = []
    retained_inputs = {} if live else {row['path']: row for row in evidence['producer_inputs']}
    if not live and len(retained_inputs) != len(evidence['producer_inputs']):
        raise ValueError('duplicate producer input snapshot')
    for path, digest in plan['inputs_sha256'].items():
        data = (_read(path, checkpoint) if live else _checked(retained_inputs[path]['snapshot'], checkpoint))
        if sha256(data).hexdigest() != digest:
            raise ValueError('historical producer input changed')
        snapshot = (_save_snapshot(snapshot_dir, 'input', data, checkpoint) if live
                    else retained_inputs[path]['snapshot'])
        inputs.append({'path': path, 'sha256': digest, 'snapshot': snapshot})
    rows = supervision.get('commands')
    expected_scan = [plan['docker_cli']['path'], 'ps', '-a', '--no-trunc', '--filter',
                     'name=^/' + plan['name'] + '$', '--format', '{{.Names}}']
    if (supervision.get('schema') != 'clay.native-container-evidence/v1'
            or supervision.get('plan_sha256') != active['plan']['sha256']
            or supervision.get('outcome') != 'FAILED' or supervision.get('baseline') is not None
            or supervision.get('cleanup_status') != 'UNVERIFIED'
            or supervision.get('container_id') is not None or supervision.get('cleanup_scans') != []
            or not isinstance(rows, list) or len(rows) != 1):
        raise ValueError('historical supervision is not a pre-create failure')
    row = rows[0]
    result = row.get('result') or {}
    if (row.get('argv') != expected_scan or row.get('stdout') != 'owned-00.stdout'
            or row.get('stderr') != 'owned-00.stderr' or result.get('job_active_after') != 0
            or result.get('timed_out') is not False
            or row.get('stdout_sha256') != sha256(out).hexdigest() or row.get('stdout_size_bytes') != len(out)
            or row.get('stderr_sha256') != sha256(err).hexdigest() or row.get('stderr_size_bytes') != len(err)):
        raise ValueError('historical baseline command evidence differs')
    if descendant_case:
        if (type(result.get('exit_code')) is not int or result['exit_code'] != 0
                or result.get('descendants_terminated') is not True
                or row.get('exception_type') != 'ValueError' or out != b'' or err != b''):
            raise ValueError('historical baseline is not the expected descendant termination')
    elif (result.get('exit_code') != 1 or result.get('descendants_terminated') is not False
          or out != b'' or b'permission denied' not in err.lower() or _ENDPOINT.encode() not in err):
        raise ValueError('historical baseline command is not the expected denied scan')
    if live and _cli_identity(plan['docker_cli']['path'], checkpoint=checkpoint) != plan['docker_cli']:
        raise ValueError('Docker CLI identity differs')
    return plan, changed, inputs


def _preparation_scope(preparation_identity, checkpoint):
    manifest = json.loads(_checked(preparation_identity, checkpoint))
    if manifest.get('schema') == 'clay.primitive-precreate-recovery-preparation/v2':
        return _DESKTOP_ENDPOINT, _SCHEMA_DESCENDANTS
    return _ENDPOINT, _SCHEMA


def _verify_scans(evidence, directory, plan, checkpoint):
    rows = evidence.get('commands')
    cli = plan['docker_cli']['path']
    endpoint = evidence['endpoint']
    scan = [cli, '-H', endpoint, 'ps', '-a', '--no-trunc', '--filter',
            'name=^/' + plan['name'] + '$', '--format', '{{.Names}}']
    expected = [[cli, '-H', endpoint, 'info', '--format', '{{.ID}}'], scan, scan,
                [cli, '-H', endpoint, 'info', '--format', '{{.ID}}']]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('recovery command denominator differs')
    outputs = []
    for i, (row, argv) in enumerate(zip(rows, expected)):
        result = row.get('result') or {}
        if (row.get('argv') != argv or result.get('exit_code') != 0
                or result.get('job_active_after') != 0 or result.get('timed_out') is not False
                or result.get('descendants_terminated') is not False):
            raise ValueError('recovery command failed or Job is not empty')
        for suffix in ('stdout', 'stderr'):
            ident = row[suffix]
            if Path(ident['path']).parent != directory or Path(ident['path']).name != f'{i:02d}.{suffix}':
                raise ValueError('recovery command output outside evidence directory')
            data = _checked(ident, checkpoint)
            if suffix == 'stdout':
                outputs.append(data)
    daemon = outputs[0].decode('utf-8').strip()
    if not daemon or daemon != outputs[3].decode('utf-8').strip() or outputs[1].strip() or outputs[2].strip():
        raise ValueError('recovery daemon identity or empty scans differ')
    if evidence.get('daemon_id') != daemon:
        raise ValueError('recorded recovery daemon identity differs')


def verify_precreate_recovery_evidence(*, admission, active, evidence_identity,
                                       preparation_identity, checkpoint):
    path = Path(evidence_identity['path']).resolve(strict=True)
    evidence = json.loads(_checked(evidence_identity, checkpoint))
    endpoint, schema = _preparation_scope(preparation_identity, checkpoint)
    if schema == _SCHEMA_DESCENDANTS and evidence.get('natural_exit_grace_seconds') != .5:
        raise ValueError('recovery natural-exit observation differs')
    if (evidence.get('schema') != schema or evidence.get('preparation') != preparation_identity
            or evidence.get('task_id') != admission.task_id
            or evidence.get('invocation_id') != active['invocation_id']
            or evidence.get('endpoint') != endpoint or evidence.get('historical_daemon_id') is not None
            or evidence.get('physical_admission') is not False):
        raise ValueError('invalid recovery evidence')
    plan, changed, inputs = _historical(admission, active, preparation_identity, checkpoint, evidence=evidence)
    if evidence.get('historical_changed_paths') != changed or evidence.get('producer_inputs') != inputs:
        raise ValueError('historical dependency changes differ')
    from clay.flow import run_admission, owned_process, native_owned_container, run_workspace, execution_budget
    from clay import api, cli
    expected_code = [Path(module.__file__).resolve() for module in
        (__import__(__name__, fromlist=['x']), run_admission, owned_process, native_owned_container,
         run_workspace, execution_budget, api, cli)]
    if [Path(row['original']['path']) for row in evidence['recovery_code']] != expected_code:
        raise ValueError('recovery code denominator differs')
    for row in evidence['recovery_code']:
        data = _checked(row['snapshot'], checkpoint)
        if sha256(data).hexdigest() != row['original']['sha256'] or len(data) != row['original']['size_bytes']:
            raise ValueError('recovery code snapshot differs')
    for row in [*changed, *inputs, *evidence['recovery_code']]:
        if Path(row['snapshot']['path']).parent != path.parent:
            raise ValueError('recovery snapshot outside evidence directory')
    _verify_scans(evidence, path.parent, plan, checkpoint)


def verify_and_scan_precreate(*, admission, active, preparation_manifest, preparation_sha256,
                              recovery_dir, budget, checkpoint):
    preparation = _identity(preparation_manifest, checkpoint)
    if preparation['sha256'] != preparation_sha256:
        raise ValueError('preparation changed before recovery')
    directory = Path(recovery_dir).resolve(strict=True)
    plan, changed, inputs = _historical(admission, active, preparation, checkpoint, snapshot_dir=directory)
    endpoint, schema = _preparation_scope(preparation, checkpoint)
    from clay.flow import run_admission, owned_process, native_owned_container, run_workspace, execution_budget
    from clay import api, cli as cli_module
    code_paths = [Path(module.__file__) for module in
        (__import__(__name__, fromlist=['x']), run_admission, owned_process, native_owned_container,
         run_workspace, execution_budget, api, cli_module)]
    code = [{'original': _identity(p, checkpoint),
             'snapshot': _save_snapshot(directory, 'code', _read(p, checkpoint), checkpoint)}
            for p in code_paths]
    for row in code:
        data = _checked(row['snapshot'], checkpoint)
        if sha256(data).hexdigest() != row['original']['sha256'] or len(data) != row['original']['size_bytes']:
            raise ValueError('recovery code changed during snapshot')
    cli = plan['docker_cli']['path']
    scan = [cli, '-H', endpoint, 'ps', '-a', '--no-trunc', '--filter',
            'name=^/' + plan['name'] + '$', '--format', '{{.Names}}']
    commands = [[cli, '-H', endpoint, 'info', '--format', '{{.ID}}'], scan, scan,
                [cli, '-H', endpoint, 'info', '--format', '{{.ID}}']]
    deadline = time.monotonic() + budget.wall_seconds
    rows = []
    for i, argv in enumerate(commands):
        checkpoint()
        if _cli_identity(cli, checkpoint=checkpoint) != plan['docker_cli']:
            raise ValueError('Docker CLI changed')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('recovery scan deadline expired')
        out, err = directory / f'{i:02d}.stdout', directory / f'{i:02d}.stderr'
        options = ({'natural_exit_grace_seconds': .5} if schema == _SCHEMA_DESCENDANTS else {})
        result = run_owned_process(argv, stdout_path=out, stderr_path=err, cwd=directory,
            timeout_seconds=remaining, checkpoint=checkpoint, cleanup_seconds=min(5.0, remaining), **options)
        rows.append({'argv': argv, 'result': asdict(result), 'stdout': _identity(out, checkpoint),
                     'stderr': _identity(err, checkpoint)})
        if result.exit_code != 0 or result.timed_out or result.descendants_terminated or result.job_active_after:
            raise ValueError('recovery read-only Docker command failed')
    daemon = _checked(rows[0]['stdout'], checkpoint).decode('utf-8').strip()
    if _cli_identity(cli, checkpoint=checkpoint) != plan['docker_cli']:
        raise ValueError('Docker CLI changed after recovery scan')
    if _historical(admission, active, preparation, checkpoint, snapshot_dir=directory) != (plan, changed, inputs):
        raise ValueError('historical live inputs changed during recovery')
    if [_identity(p, checkpoint) for p in code_paths] != [row['original'] for row in code]:
        raise ValueError('recovery code changed during scan')
    evidence = {'schema': schema, 'task_id': admission.task_id, 'invocation_id': active['invocation_id'],
                'preparation': preparation, 'endpoint': endpoint, 'daemon_id': daemon,
                'historical_daemon_id': None, 'historical_changed_paths': changed, 'producer_inputs': inputs,
                'commands': rows, 'recovery_code': code,
                'physical_admission': False}
    if schema == _SCHEMA_DESCENDANTS:
        evidence['natural_exit_grace_seconds'] = .5
    path = directory / 'evidence.json'
    atomic_json(path, evidence)
    identity = _identity(path, checkpoint)
    verify_precreate_recovery_evidence(admission=admission, active=active, evidence_identity=identity,
                                       preparation_identity=preparation, checkpoint=checkpoint)
    return identity
