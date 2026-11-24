"""Bound Docker lifecycle for the optional native primitive observation.

The caller retains task admission and source authority. Docker is a separate
daemon: a Job-owned CLI exit is never treated as container cleanup.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import ntpath
import itertools
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

from clay.flow.owned_process import run_owned_process
from clay.flow.control_stop import record_unwind_failure
from clay.flow.run_workspace import atomic_json

_MAX_LOG_BYTES = 64 * 1024 * 1024
_MAX_REPLY_BYTES = 1024 * 1024


def _hash_file(path, *, checkpoint=lambda: None, max_bytes=_MAX_LOG_BYTES):
    digest, size = sha256(), 0
    with Path(path).open('rb') as stream:
        while True:
            checkpoint()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError('native command output exceeds byte limit')
            digest.update(chunk)
    return digest.hexdigest(), size


def _digest(path, *, checkpoint=lambda: None, max_bytes=_MAX_LOG_BYTES):
    return _hash_file(path, checkpoint=checkpoint, max_bytes=max_bytes)[0]


def _read_bounded(path, *, limit=_MAX_REPLY_BYTES):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('native command reply exceeds byte limit')
    return data


def _cli_identity(path, *, checkpoint):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file():
        raise ValueError('Docker CLI is not a regular file')
    digest = _digest(path, checkpoint=checkpoint, max_bytes=256 * 1024 * 1024)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Docker CLI changed during identity observation')
    return {'path': str(path), 'sha256': digest, 'size_bytes': after.st_size}


def _mount(value):
    value = str(value).replace('\\', '/')
    for prefix in ('/run/desktop/mnt/host/', '/host_mnt/'):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            if len(value) >= 2 and value[0].isalpha() and value[1] == '/':
                value = value[0]+':'+value[1:]
    if len(value) >= 3 and value[0] == '/' and value[1].isalpha() and value[2] == '/':
        value = value[1]+':'+value[2:]
    if not ntpath.isabs(value):
        raise ValueError('unrecognized Docker bind path')
    return ntpath.normcase(ntpath.normpath(value))


def _inspect_owned(value, plan):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError('native container inspection is not unique')
    item = value[0]
    config, host = item['Config'], item['HostConfig']
    mounts = item.get('Mounts', [])
    if (item['Name'].lstrip('/') != plan['name'] or item['Image'] != plan['image']
            or config['Image'] != plan['image']
            or (config.get('Labels') or {}).get('org.clay.primitive-nonce') != plan['nonce']
            or host.get('NetworkMode') != 'none' or host.get('ReadonlyRootfs') is not True
            or host.get('Tmpfs') != plan['tmpfs']
            or len(mounts) != 1 or mounts[0].get('Type') != 'bind'
            or mounts[0].get('Destination') != '/work' or mounts[0].get('RW') is not True
            or _mount(mounts[0]['Source']) != _mount(plan['work_dir'])):
        raise ValueError('native container identity or isolation differs')
    return item


def run_native_container(*, stage, work_dir, image, runtime_sha256, pdk_root,
                         timeout_seconds, checkpoint, inputs_sha256, runtime_contract):
    """Execute the existing /work/generate.py under one inspected container."""
    work = Path(work_dir).resolve(strict=True)
    if (not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or not image.startswith('sha256:') or len(image) != 71):
        raise ValueError('native container requires finite budget and image ID')
    if (type(runtime_contract) is not dict or runtime_contract.get('image') != image
            or runtime_contract.get('pdk_root') != pdk_root
            or timeout_seconds > runtime_contract.get('timeout_seconds', 0)):
        raise ValueError('native container runtime contract differs')
    runtime_contract_sha256 = sha256(json.dumps(runtime_contract, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if (type(inputs_sha256) is not dict or not inputs_sha256
            or any(type(path) is not str or type(digest) is not str
                   or not re.fullmatch(r'[0-9a-f]{64}', digest)
                   or not Path(path).is_absolute() or str(Path(path).resolve()) != path
                   or not Path(path).is_relative_to(work)
                   for path, digest in inputs_sha256.items())):
        raise ValueError('native producer inputs require canonical paths and SHA256 values')
    for path, digest in inputs_sha256.items():
        if _digest(path, checkpoint=checkpoint) != digest:
            raise ValueError('native producer input changed before launch')
    cli_path = shutil.which('docker')
    if cli_path is None:
        raise FileNotFoundError('Docker CLI not found')
    cli = _cli_identity(cli_path, checkpoint=checkpoint)
    plan = {'schema': 'clay.native-container-plan/v1', 'invocation_id': stage.invocation_id,
            'name': 'clay-primitive-'+stage.invocation_id, 'nonce': stage.invocation_id,
            'image': image, 'work_dir': str(work), 'runtime_sha256': runtime_sha256, 'pdk_root': pdk_root,
            'network': 'none', 'read_only_root': True, 'timeout_seconds': timeout_seconds,
            'cleanup_seconds': min(30.0, timeout_seconds), 'inputs_sha256': dict(inputs_sha256),
            'docker_cli': cli, 'runtime_contract_sha256': runtime_contract_sha256, 'tmpfs': {'/tmp': 'rw'}}
    request = getattr(stage, 'role_trace_request', None)
    if request is not None:
        plan['schema'] = 'clay.native-container-plan/v2'
        plan['role_trace_request'] = request
    plan_identity = stage.bind_resource_plan(plan)
    deadline = time.monotonic()+timeout_seconds
    cleanup_deadline = None
    commands = []
    sequence = itertools.count()

    def command(args, *, allow_failure=False, cleanup=False, read_output=True):
        remaining = (cleanup_deadline if cleanup else deadline)-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('native container total deadline expired')
        index = next(sequence)
        out, err = work/f'owned-{index:02d}.stdout', work/f'owned-{index:02d}.stderr'
        row = {'argv': [cli['path'], *args], 'stdout': out.name, 'stderr': err.name,
               'result': None, 'stdout_sha256': None, 'stderr_sha256': None,
               'stdout_size_bytes': None, 'stderr_size_bytes': None}
        commands.append(row)
        def progress():
            if cleanup:
                if time.monotonic() >= cleanup_deadline:
                    raise TimeoutError('native container cleanup deadline expired')
            else:
                checkpoint()
                if time.monotonic() >= deadline:
                    raise TimeoutError('native container total deadline expired')
        def record_failure(error):
            row['exception_type'] = type(error).__name__
            evidence_deadline = time.monotonic() + min(5.0, plan['cleanup_seconds'])
            def evidence_checkpoint():
                if time.monotonic() >= evidence_deadline:
                    raise TimeoutError('native command evidence deadline expired')
            for name, path in (('stdout', out), ('stderr', err)):
                try:
                    if row[name+'_sha256'] is None and path.is_file():
                        row[name+'_sha256'], row[name+'_size_bytes'] = _hash_file(
                            path, checkpoint=evidence_checkpoint)
                except BaseException as ending:
                    record_unwind_failure(error, ending)
        try:
            if _cli_identity(cli['path'], checkpoint=progress) != cli:
                raise ValueError('Docker CLI identity changed before command')
            result = run_owned_process([cli['path'], *args], stdout_path=out, stderr_path=err,
                timeout_seconds=remaining, checkpoint=None if cleanup else checkpoint,
                cwd=work, cleanup_seconds=min(5.0, remaining), natural_exit_grace_seconds=.5)
            row['result'] = asdict(result)
            if _cli_identity(cli['path'], checkpoint=progress) != cli:
                raise ValueError('Docker CLI identity changed after command')
        except BaseException as error:
            if row['result'] is None:
                row['result'] = getattr(error, 'owned_process', None)
            record_failure(error)
            raise
        try:
            if (result.job_active_after != 0 or result.timed_out or result.descendants_terminated
                    or (result.exit_code and not allow_failure)):
                raise ValueError('owned Docker command did not complete: '+args[0])
            if read_output:
                progress()
                reply = _read_bounded(out)
                row['stdout_sha256'] = sha256(reply).hexdigest()
                row['stdout_size_bytes'] = len(reply)
                progress()
            else:
                reply = b''
                row['stdout_sha256'], row['stdout_size_bytes'] = _hash_file(out, checkpoint=progress)
            row['stderr_sha256'], row['stderr_size_bytes'] = _hash_file(err, checkpoint=progress)
            return reply
        except BaseException as error:
            record_failure(error)
            raise

    def scan(*, cleanup=False):
        output = command(['ps', '-a', '--no-trunc', '--filter', 'name=^/'+plan['name']+'$',
                          '--format', '{{.Names}}'], cleanup=cleanup).decode('utf-8').splitlines()
        return [name for name in output if name == plan['name']]

    def inspect(ident, *, cleanup=False):
        return _inspect_owned(json.loads(command(['inspect', ident], cleanup=cleanup)), plan)

    baseline = None
    container_id = None
    actual_platform = None
    primary = None
    cleanup_scans = []
    create_attempted = False
    try:
        baseline = scan()
        if baseline:
            raise ValueError('native container name already exists')
        create_attempted = True
        raw = command(['create', '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw',
            '--name', plan['name'], '--label', 'org.clay.primitive-nonce='+plan['nonce'],
            '-v', work.as_posix()+':/work', '-w', '/work', '-e', 'PDK=sky130A',
            '-e', 'PDK_ROOT='+pdk_root, '-e', 'PDKPATH='+pdk_root+'/sky130A',
            '--entrypoint', 'bash', image, '-lc', 'exec python3 /work/generate.py'])
        container_id = raw.decode('ascii').strip()
        if len(container_id) != 64 or any(c not in '0123456789abcdef' for c in container_id):
            raise ValueError('Docker did not return a full container ID')
        item = inspect(container_id)
        if item['Id'] != container_id:
            raise ValueError('created native container ID differs')
        images = json.loads(command(['image', 'inspect', image]))
        if (not isinstance(images, list) or len(images) != 1 or images[0]['Id'] != image
                or not images[0].get('Os') or not images[0].get('Architecture')):
            raise ValueError('actual native image identity or platform differs')
        actual_platform = {'os': images[0]['Os'], 'architecture': images[0]['Architecture']}
        command(['start', '--attach', container_id], read_output=False)
        item = inspect(container_id)
        if item['Id'] != container_id or item['State']['Running'] or item['State']['ExitCode'] != 0:
            raise ValueError('native container did not exit successfully')
    except BaseException as error:
        primary = error
    cleanup_deadline = time.monotonic()+plan['cleanup_seconds']
    try:
        if baseline is None:
            raise ValueError('native container baseline unverified')
        if baseline:
            raise ValueError('native container baseline collision')
        before = scan(cleanup=True)
        cleanup_scans.append(before)
        if create_attempted and (container_id is None or not re.fullmatch(r'[0-9a-f]{64}', container_id)):
            raise ValueError('refusing to assume cleanup without a captured full container ID')
        if container_id is not None:
            item = inspect(container_id, cleanup=True)
            if item['Id'] != container_id:
                raise ValueError('refusing to remove a different container')
            command(['rm', '-f', container_id], cleanup=True)
        elif before:
            raise ValueError('refusing to remove an unowned container')
        for _ in range(2):
            cleanup_scans.append(scan(cleanup=True))
        if cleanup_scans[-2:] != [[], []]:
            raise ValueError('native container cleanup is unverified')
    except BaseException as cleanup_error:
        if primary is not None:
            record_unwind_failure(primary, cleanup_error)
        else:
            primary = cleanup_error
        # Python clears the exception target at the end of the except block.
        cleanup_failed = True
    else:
        cleanup_failed = False
    evidence = {'schema': 'clay.native-container-evidence/v1', 'plan_sha256': plan_identity['sha256'],
                'container_id': container_id, 'actual_platform': actual_platform,
                'baseline': baseline, 'cleanup_scans': cleanup_scans, 'commands': commands,
                'cleanup_status': 'UNVERIFIED' if cleanup_failed else 'CLEAN',
                'outcome': 'FAILED' if primary else 'COMPLETE'}
    evidence_path = work/'native-supervision.json'
    try:
        atomic_json(evidence_path, evidence)
    except BaseException as ending:
        if primary is not None:
            record_unwind_failure(primary, ending)
            raise primary
        raise
    if primary is not None:
        try:
            primary.native_container_evidence_path = str(evidence_path)
            BaseException.add_note(primary, 'native container failure evidence: '+str(evidence_path))
        except BaseException as ending:
            record_unwind_failure(primary, ending)
        raise primary
    return evidence_path, plan, plan_identity


def replay_native_container_evidence(evidence_path, *, plan, plan_identity, checkpoint, expected_identity):
    """Read retained supervisor records without launching a new Docker command."""
    path = Path(evidence_path).resolve(strict=True)
    plan_path = Path(plan_identity['path']).resolve(strict=True)
    checkpoint()
    plan_bytes = _read_bounded(plan_path)
    if (sha256(plan_bytes).hexdigest() != plan_identity['sha256']
            or len(plan_bytes) != plan_identity['size_bytes']
            or json.loads(plan_bytes) != plan):
        raise ValueError('native container plan changed')
    cli = plan['docker_cli']
    if (not isinstance(cli, dict) or set(cli) != {'path', 'sha256', 'size_bytes'}
            or not isinstance(cli['path'], str) or not Path(cli['path']).is_absolute()
            or not isinstance(cli['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', cli['sha256'])
            or type(cli['size_bytes']) is not int or cli['size_bytes'] <= 0):
        raise ValueError('observed Docker CLI identity is invalid')
    evidence_bytes = _read_bounded(path)
    if (not isinstance(expected_identity, dict) or expected_identity.get('path') != str(path)
            or expected_identity.get('sha256') != sha256(evidence_bytes).hexdigest()
            or expected_identity.get('size_bytes') != len(evidence_bytes)):
        raise ValueError('native container evidence identity differs')
    data = json.loads(evidence_bytes)
    if (data['schema'] != 'clay.native-container-evidence/v1'
            or data['plan_sha256'] != plan_identity['sha256'] or data['baseline'] != []
            or data['outcome'] not in ('COMPLETE', 'FAILED') or data['cleanup_status'] != 'CLEAN'
            or data['cleanup_scans'][-2:] != [[], []]):
        raise ValueError('native container evidence is incomplete')
    if not isinstance(data['commands'], list):
        raise ValueError('native command transcript is invalid')
    parsed_stdout = {}
    for index, row in enumerate(data['commands']):
        checkpoint()
        stdout, stderr = f'owned-{index:02d}.stdout', f'owned-{index:02d}.stderr'
        if (not isinstance(row, dict) or row.get('stdout') != stdout or row.get('stderr') != stderr
                or (path.parent/stdout).resolve().parent != path.parent
                or (path.parent/stderr).resolve().parent != path.parent
                or not isinstance(row.get('result'), dict) or row['result'].get('job_active_after') != 0
                or row['result'].get('cleanup_status', 'CLEAN') != 'CLEAN'):
            raise ValueError('native Docker command evidence changed')
        argv = row.get('argv')
        if not isinstance(argv, list) or len(argv) < 2:
            raise ValueError('native Docker command evidence changed')
        parsed = argv[1] in {'ps', 'create', 'inspect'} or argv[1:3] == ['image', 'inspect']
        if parsed:
            checkpoint()
            reply = _read_bounded(path.parent/stdout)
            stdout_sha = sha256(reply).hexdigest()
            stdout_size = len(reply)
            parsed_stdout[stdout] = reply
        else:
            stdout_sha, stdout_size = _hash_file(path.parent/stdout, checkpoint=checkpoint)
        stderr_sha, stderr_size = _hash_file(path.parent/stderr, checkpoint=checkpoint)
        if (stdout_sha != row['stdout_sha256'] or stdout_size != row['stdout_size_bytes']
                or stderr_sha != row['stderr_sha256'] or stderr_size != row['stderr_size_bytes']):
            raise ValueError('native Docker command evidence changed')
    rows = data['commands']
    scans = [row for row in rows if row['argv'][1:2] == ['ps']]
    expected_scan = [cli['path'], 'ps', '-a', '--no-trunc', '--filter',
                     'name=^/'+plan['name']+'$', '--format', '{{.Names}}']
    if len(scans) != 4 or any(row['argv'] != expected_scan for row in scans):
        raise ValueError('native container scan denominator differs')
    observed_scans = []
    for row in scans:
        if row['result']['exit_code'] != 0 or row['result']['timed_out']:
            raise ValueError('native container scan failed')
        names = parsed_stdout[row['stdout']].decode('utf-8').splitlines()
        observed_scans.append([name for name in names if name == plan['name']])
    if observed_scans != [data['baseline'], *data['cleanup_scans']]:
        raise ValueError('native container scan transcript differs')
    creates = [row for row in rows if row['argv'][1:2] == ['create']]
    inspections = [row for row in rows if row['argv'][1:2] == ['inspect']]
    images = [row for row in rows if row['argv'][1:3] == ['image', 'inspect']]
    starts = [row for row in rows if row['argv'][1:3] == ['start', '--attach']]
    if data['outcome'] == 'COMPLETE' and (len(creates) != 1 or len(inspections) < 2
            or len(images) != 1 or len(starts) != 1 or not data['container_id'] or not data['actual_platform']
            or any(row['result']['exit_code'] != 0 or row['result']['timed_out']
                   or row['result']['descendants_terminated'] for row in rows)):
        raise ValueError('native container command denominator differs')
    if creates and data['container_id'] is not None:
        if len(creates) != 1 or parsed_stdout[creates[0]['stdout']].decode('ascii').strip() != data['container_id']:
            raise ValueError('native created container identity differs')
    expected_create = [cli['path'], 'create', '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw',
        '--name', plan['name'], '--label', 'org.clay.primitive-nonce='+plan['nonce'],
        '-v', Path(plan['work_dir']).as_posix()+':/work', '-w', '/work', '-e', 'PDK=sky130A',
        '-e', 'PDK_ROOT='+plan['pdk_root'], '-e', 'PDKPATH='+plan['pdk_root']+'/sky130A',
        '--entrypoint', 'bash', plan['image'], '-lc', 'exec python3 /work/generate.py']
    if any(row['argv'] != expected_create for row in creates):
        raise ValueError('native create command differs')
    if any(row['argv'] != [cli['path'], 'inspect', data['container_id']] for row in inspections):
        raise ValueError('native inspection target differs')
    if any(row['argv'] != [cli['path'], 'start', '--attach', data['container_id']] for row in starts):
        raise ValueError('native start target differs')
    removals = [row for row in rows if row['argv'][1:2] == ['rm']]
    if any(row['argv'] != [cli['path'], 'rm', '-f', data['container_id']] for row in removals):
        raise ValueError('native removal target differs')
    expected_image = [cli['path'], 'image', 'inspect', plan['image']]
    if any(row['argv'] != expected_image for row in images):
        raise ValueError('native image inspection target differs')
    if data['outcome'] == 'COMPLETE':
        expected_order = [expected_scan, expected_create,
                          [cli['path'], 'inspect', data['container_id']], expected_image,
                          [cli['path'], 'start', '--attach', data['container_id']],
                          [cli['path'], 'inspect', data['container_id']], expected_scan,
                          [cli['path'], 'inspect', data['container_id']],
                          [cli['path'], 'rm', '-f', data['container_id']], expected_scan, expected_scan]
        if [row['argv'] for row in rows] != expected_order:
            raise ValueError('native container command order differs')
    for row in inspections:
        actual = _inspect_owned(json.loads(parsed_stdout[row['stdout']]), plan)
        if actual['Id'] != data['container_id']:
            raise ValueError('native container inspection ID differs')
    if data['outcome'] == 'COMPLETE':
        finished = json.loads(parsed_stdout[inspections[1]['stdout']])[0]['State']
        if finished['Running'] or finished['ExitCode'] != 0:
            raise ValueError('native container completion state differs')
    if images:
        if len(images) != 1:
            raise ValueError('native image inspection denominator differs')
        image_rows = json.loads(parsed_stdout[images[0]['stdout']])
        if (len(image_rows) != 1 or image_rows[0]['Id'] != plan['image']
                or data['actual_platform'] != {'os': image_rows[0]['Os'],
                                               'architecture': image_rows[0]['Architecture']}):
            raise ValueError('native image platform evidence differs')
    return data


def verify_native_container_evidence(evidence_path, *, plan, plan_identity, checkpoint, expected_identity):
    """Validate retained records and perform the live cleanup recheck."""
    cli = plan['docker_cli']
    if _cli_identity(cli['path'], checkpoint=checkpoint) != cli:
        raise ValueError('observed Docker CLI identity changed')
    data = replay_native_container_evidence(evidence_path, plan=plan, plan_identity=plan_identity,
                                             checkpoint=checkpoint, expected_identity=expected_identity)
    path = Path(evidence_path).resolve(strict=True)
    plan_path = Path(plan_identity['path']).resolve(strict=True)
    if _cli_identity(cli['path'], checkpoint=checkpoint) != cli:
        raise ValueError('observed Docker CLI identity changed')
    verification = plan_path.parent/('container-recheck-'+uuid4().hex)
    out, err = verification.with_suffix('.stdout'), verification.with_suffix('.stderr')
    result = run_owned_process([cli['path'], 'ps', '-a', '--no-trunc', '--filter',
        'name=^/'+plan['name']+'$', '--format', '{{.Names}}'], stdout_path=out, stderr_path=err,
        timeout_seconds=min(10.0, plan['cleanup_seconds']), checkpoint=checkpoint,
        cwd=path.parent, cleanup_seconds=min(5.0, plan['cleanup_seconds']),
        natural_exit_grace_seconds=.5)
    if (result.exit_code != 0 or result.timed_out or result.descendants_terminated
            or result.job_active_after != 0 or _read_bounded(out).strip()):
        raise ValueError('native container final cleanup recheck failed')
    if _cli_identity(cli['path'], checkpoint=checkpoint) != cli:
        raise ValueError('observed Docker CLI identity changed during cleanup recheck')
    return data
