"""Source-bound receipts for an optional PDK execution-frame observation.

The retained record is an observation of one native producer invocation. It
does not map source nets or diffusion sides, certify body or cuts, or admit a
physical stage. All executable/PDK runtime identity limits remain explicit.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re

_MANIFEST = 'role-trace.json'
_LIMIT = 256 * 1024 * 1024
_UNKNOWN = (
    'SOURCE_ROLE_BINDING', 'SOURCE_TERMINAL_SIDE_BINDING', 'SOURCE_NET_BINDING',
    'SOURCE_BODY_NET_BINDING', 'BODY_SEMICONDUCTOR_COVERAGE', 'CUT_GEOMETRY_COMPLETENESS',
    'PRIMITIVE_STAGE_ADMISSION', 'RESOURCE_SUPERVISION_FRESHNESS',
    'INPUT_CONSUMPTION_IDENTITY', 'LOADED_CODE_IDENTITY', 'LOADED_DOCKER_CLI_IDENTITY',
    'TOOLKIT_SOURCE_EXPECTATION',
)


def _read(path, checkpoint, limit=_LIMIT):
    path = Path(path)
    chunks, size = [], 0
    with path.open('rb') as stream:
        while True:
            checkpoint()
            block = stream.read(min(1024 * 1024, limit + 1 - size))
            if not block:
                break
            size += len(block)
            if size > limit:
                raise ValueError('role trace input exceeds byte limit')
            chunks.append(block)
    checkpoint()
    return b''.join(chunks)


def _identity(path, checkpoint):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('role trace file path is not a unique regular file')
    data = _read(path, checkpoint)
    return {'path': str(path), 'sha256': sha256(data).hexdigest(), 'size_bytes': len(data)}


def _read_identity(identity, checkpoint):
    if (not isinstance(identity, dict) or set(identity) != {'path', 'sha256', 'size_bytes'}
            or not isinstance(identity.get('path'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', str(identity.get('sha256')))
            or type(identity.get('size_bytes')) is not int or not 0 <= identity['size_bytes'] <= _LIMIT):
        raise ValueError('invalid role trace file identity')
    path = Path(identity['path'])
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('role trace file identity path differs')
    data = _read(path, checkpoint, limit=identity['size_bytes'])
    if len(data) != identity['size_bytes'] or sha256(data).hexdigest() != identity['sha256']:
        raise ValueError('role trace bound bytes changed')
    return data


def _request(request):
    from clay.flow.run_admission import _checked_role_trace_request
    if not isinstance(request, dict):
        raise ValueError('role trace request missing')
    return _checked_role_trace_request(request, {'wall_seconds': request.get('wall_seconds')})


def _inside(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('invalid role trace artifact path')
    path = root / relative
    if path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path:
        return path
    raise ValueError('role trace artifact escapes its output root')


def _artifact(root, relative, checkpoint):
    path = _inside(root, relative)
    identity = _identity(path, checkpoint)
    return {'sha256': identity['sha256'], 'size_bytes': identity['size_bytes']}


def _artifact_bytes(root, relative, checkpoint, limit=_LIMIT):
    path = _inside(root, relative)
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('role trace artifact is not a unique regular file')
    data = _read(path, checkpoint, limit=limit)
    return data, {'sha256': sha256(data).hexdigest(), 'size_bytes': len(data)}


def _executions(data, *, keys, suffix, pdk_root):
    rows = json.loads(data)
    if not isinstance(rows, list) or len(rows) != len(keys):
        raise ValueError('role trace Magic execution denominator differs')
    rc = pdk_root.rstrip('/') + '/sky130A/libs.tech/magic/sky130A.magicrc'
    for row, key in zip(rows, sorted(keys)):
        expected = ['magic', '-dnull', '-noconsole', '-rcfile', rc,
                    '/work/' + key + suffix + '/generate.tcl']
        if (not isinstance(row, dict) or row.get('key') != key or row.get('command') != expected
                or type(row.get('returncode')) is not int or row['returncode'] != 0
                or type(row.get('seconds')) not in (int, float) or not math.isfinite(row['seconds'])
                or row['seconds'] < 0):
            raise ValueError('role trace Magic execution differs')
    return rows


def _source_specs(source_bytes, source_identity):
    from clay.layout.native_sky130 import _cell_native_specs, _digest, native_source_identity
    from clay.parser import parse_clay_source
    parsed = parse_clay_source(source_bytes.decode('utf-8-sig'))
    cells = [cell for cell in parsed.cell_defs if native_source_identity(cell) == source_identity]
    if len(cells) != 1:
        raise ValueError('role trace source identity does not select one cell')
    source_specs = _cell_native_specs(cells[0])
    if not source_specs:
        raise ValueError('role trace source contains no native members')
    specs, members = {}, {}
    for name, spec in source_specs.items():
        key = 'pcell_' + _digest(spec)[:20]
        if key in specs and specs[key] != spec:
            raise ValueError('role trace truncated source spec key collision')
        specs[key] = spec
        members[name] = key
    return specs, members


def build_role_trace_manifest(*, output, specs, members, request, invocation_id,
                              source_identity, runtime_sha256, runtime_file_identity, plan_identity,
                              evidence_identity, checkpoint):
    """Recompute a capability observation from the source producer and retained files."""
    from clay.flow.native_owned_container import replay_native_container_evidence
    from clay.flow.primitive_role_trace import (eligible_spec, instrument_generation_tcl,
        verify_trace_stdout, compare_artifacts, observe_label_components)
    from clay.flow.primitive_mim_role_trace import (eligible_for_request, instrument_for_request,
        verify_mim_frames, observe_mim_components)
    from clay.layout.native_sky130 import _digest, _generation_tcl, _runner_script
    from clay.verification.native_sky130 import _runtime_expectations, _toolchain_script
    from clay.flow.run_admission import _runtime_contract_sha256

    _request(request)
    if (not re.fullmatch(r'[0-9a-f]{32}', str(invocation_id))
            or not re.fullmatch(r'[0-9a-f]{64}', str(source_identity))
            or not re.fullmatch(r'[0-9a-f]{64}', str(runtime_sha256))):
        raise ValueError('invalid role trace invocation or source identity')
    root = Path(output).absolute()
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError('invalid role trace output root')
    plan_path = Path(plan_identity['path'])
    if plan_path.name != 'native-container-plan.json' or plan_path.parent.name != invocation_id:
        raise ValueError('role trace plan belongs to another invocation')
    plan = json.loads(_read_identity(plan_identity, checkpoint))
    runtime_bytes = _read_identity(runtime_file_identity, checkpoint)
    if runtime_file_identity['sha256'] != runtime_sha256:
        raise ValueError('role trace source runtime identity differs')
    runtime = json.loads(runtime_bytes)
    if (plan.get('schema') != 'clay.native-container-plan/v2' or plan.get('invocation_id') != invocation_id
            or plan.get('work_dir') != str(root) or plan.get('runtime_sha256') != runtime_sha256
            or plan.get('role_trace_request') != request
            or type(plan.get('timeout_seconds')) not in (int, float)
            or not 0 < plan['timeout_seconds'] <= request['wall_seconds']
            or not isinstance(runtime, dict) or runtime.get('schema') != 'clay.native-physical-runtime/v1'
            or runtime.get('image') != plan.get('image') or runtime.get('pdk_root') != plan.get('pdk_root')
            or type(runtime.get('timeout_seconds')) not in (int, float)
            or plan['timeout_seconds'] > runtime['timeout_seconds']
            or plan.get('runtime_contract_sha256') != _runtime_contract_sha256(runtime)):
        raise ValueError('role trace plan request or runtime differs')
    if Path(evidence_identity['path']) != root / 'native-supervision.json':
        raise ValueError('role trace supervisor belongs to another producer')
    resource = replay_native_container_evidence(evidence_identity['path'], plan=plan,
        plan_identity=plan_identity, checkpoint=checkpoint, expected_identity=evidence_identity)
    if (resource.get('outcome') != 'COMPLETE' or resource.get('cleanup_status') != 'CLEAN'
            or resource.get('baseline') != [] or resource.get('cleanup_scans', [])[-2:] != [[], []]):
        raise ValueError('role trace producer execution is incomplete')
    if (not isinstance(specs, dict) or not specs or not isinstance(members, dict) or not members
            or any(not isinstance(name, str) or not name or key not in specs for name, key in members.items())
            or set(members.values()) != set(specs)):
        raise ValueError('role trace source member coverage differs')
    for key, spec in specs.items():
        if key != 'pcell_' + _digest(spec)[:20]:
            raise ValueError('role trace source specification key differs')
        if type(spec.get('occurrences')) is not int or spec['occurrences'] < 1:
            raise ValueError('role trace source occurrence denominator differs')
    actual_dirs = {path.name for path in root.iterdir() if path.name.startswith('pcell_') and path.is_dir()}
    if actual_dirs != set(specs):
        raise ValueError('role trace producer master denominator differs')
    eligible = sorted(key for key, spec in specs.items() if eligible_for_request(spec, request))
    runner_bytes, runner_identity = _artifact_bytes(root, 'generate.py', checkpoint, limit=16 * 1024 * 1024)
    expected_runner = _runner_script(specs, plan['pdk_root'], role_trace_keys=eligible, role_trace_enabled=True)
    if runner_bytes != expected_runner.encode('utf-8'):
        raise ValueError('role trace runner differs from native producer')
    toolchain_bytes, toolchain_identity = _artifact_bytes(root, 'toolchain.py', checkpoint, limit=16 * 1024 * 1024)
    contracts = {'models': {spec['model']: spec['contract'] for spec in specs.values()}}
    expected_files = _runtime_expectations(runtime.get('files'), plan['pdk_root'])
    expected_toolchain = _toolchain_script(plan['pdk_root'], contracts, expected_files)
    if toolchain_bytes != expected_toolchain.encode('utf-8'):
        raise ValueError('role trace toolchain script differs from source runtime')
    execution_bytes, execution_identity = _artifact_bytes(root, 'executions.json', checkpoint,
                                                            limit=16 * 1024 * 1024)
    _executions(execution_bytes, keys=specs, suffix='', pdk_root=plan['pdk_root'])
    if eligible:
        trace_exec_bytes, trace_exec_identity = _artifact_bytes(root, 'role-trace-executions.json', checkpoint,
                                                                limit=16 * 1024 * 1024)
        _executions(trace_exec_bytes, keys=eligible, suffix='/trace', pdk_root=plan['pdk_root'])
    elif (root / 'role-trace-executions.json').exists():
        raise ValueError('unsupported trace run has unexpected Magic executions')
    lyp_bytes, lyp_identity = _artifact_bytes(root, 'role-trace.sky130A.lyp', checkpoint, limit=16 * 1024 * 1024)
    lyp_runtime_bytes, lyp_runtime_identity = _artifact_bytes(root, 'role-trace-runtime.json', checkpoint, limit=1024 * 1024)
    lyp_runtime = json.loads(lyp_runtime_bytes)
    expected_lyp = plan['pdk_root'].rstrip('/') + '/sky130A/libs.tech/klayout/tech/sky130A.lyp'
    if (lyp_runtime != {'path': expected_lyp, 'sha256': sha256(lyp_bytes).hexdigest(),
                        'size_bytes': len(lyp_bytes)}):
        raise ValueError('role trace PDK layer-purpose runtime observation differs')
    toolkit_bytes, toolkit_identity = _artifact_bytes(root, 'pcell-runtime.json', checkpoint, limit=1024 * 1024)
    toolkit = json.loads(toolkit_bytes)
    if (not isinstance(toolkit, dict) or not isinstance(toolkit.get('path'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', str(toolkit.get('sha256')))):
        raise ValueError('role trace toolkit runtime observation is invalid')
    file_identities = {'generate.py': runner_identity, 'toolchain.py': toolchain_identity,
        'executions.json': execution_identity, 'pcell-runtime.json': toolkit_identity,
        'toolchain.json': _artifact(root, 'toolchain.json', checkpoint),
        'role-trace.sky130A.lyp': lyp_identity, 'role-trace-runtime.json': lyp_runtime_identity}
    if eligible:
        file_identities['role-trace-executions.json'] = trace_exec_identity
    key_results = {}
    for key, spec in sorted(specs.items()):
        checkpoint()
        folder = _inside(root, key)
        request_bytes, request_identity = _artifact_bytes(root, f'{key}/request.json', checkpoint, limit=16 * 1024 * 1024)
        if json.loads(request_bytes) != spec:
            raise ValueError('role trace generated request differs from source spec')
        control_bytes, control_identity = _artifact_bytes(root, f'{key}/generate.tcl', checkpoint, limit=16 * 1024 * 1024)
        if control_bytes.decode('utf-8') != _generation_tcl(spec):
            raise ValueError('role trace control script differs from native producer')
        source_files = [f'{key}/request.json', f'{key}/generate.tcl']
        file_identities[source_files[0]], file_identities[source_files[1]] = request_identity, control_identity
        record = {'source_spec_sha256': _digest(spec), 'source_occurrences': spec['occurrences'],
                  'eligibility': 'ELIGIBLE' if key in eligible else 'UNSUPPORTED',
                  'status': 'UNKNOWN', 'physical_admission': False}
        if key not in eligible:
            record['reason'] = ('SPEC_OUTSIDE_ROLE_TRACE_V1' if request['schema'].endswith('/v1')
                                else 'SPEC_OUTSIDE_ROLE_TRACE_V2')
            if (folder / 'trace').exists():
                raise ValueError('unsupported source master has trace output')
        else:
            trace = _inside(root, f'{key}/trace')
            trace_script, trace_identity = _artifact_bytes(root, f'{key}/trace/generate.tcl', checkpoint, limit=16 * 1024 * 1024)
            if trace_script.decode('utf-8') != instrument_for_request(spec, control_bytes.decode('utf-8'), request):
                raise ValueError('role trace instrumentation differs from native generator')
            source_files += [f'{key}/trace/generate.tcl']
            file_identities[source_files[-1]] = trace_identity
            for name in ('stdout.log', 'stderr.log', 'primitive.gds', 'primitive_flat.ext',
                         'native.spice', 'drc.txt', 'scale_um.txt'):
                source_files.extend((f'{key}/{name}', f'{key}/trace/{name}'))
            # Bind all paired files before parsing or comparing. Invalid evidence
            # remains UNKNOWN and can never be promoted into a frame observation.
            for relative in source_files:
                if relative not in file_identities:
                    file_identities[relative] = _artifact(root, relative, checkpoint)
            try:
                noninterference = compare_artifacts(folder, trace, checkpoint)
                stdout, stdout_identity = _artifact_bytes(root, f'{key}/trace/stdout.log', checkpoint, limit=16 * 1024 * 1024)
                if stdout_identity != file_identities[f'{key}/trace/stdout.log']:
                    raise ValueError('role trace parsed stdout identity differs')
                if spec['family'] == 'mim':
                    frames = verify_mim_frames(stdout)
                    nodes = observe_mim_components(trace, lyp_bytes, stdout, checkpoint)
                else:
                    frames = verify_trace_stdout(stdout)
                    nodes = observe_label_components(trace, lyp_bytes, stdout, checkpoint)
                for filename, identities in noninterference['files'].items():
                    if (file_identities[f'{key}/{filename}'] != {'sha256': identities['control_sha256'], 'size_bytes': identities['control_size_bytes']}
                            or file_identities[f'{key}/trace/{filename}'] != {'sha256': identities['trace_sha256'], 'size_bytes': identities['trace_size_bytes']}):
                        raise ValueError('role trace compared bytes differ from capture')
                if (nodes['gds_observed'] != file_identities[f'{key}/trace/primitive.gds']
                        or nodes['scale_observed'] != file_identities[f'{key}/trace/scale_um.txt']
                        or nodes['trace_stdout_observed'] != stdout_identity
                        or nodes['lyp_observed_sha256'] != lyp_identity['sha256']
                        or nodes['lyp_observed_size_bytes'] != lyp_identity['size_bytes']):
                    raise ValueError('role trace label observation bytes differ from capture')
            except (ValueError, UnicodeError, FileNotFoundError, OSError):
                record['reason'] = 'ROLE_TRACE_EVIDENCE_UNRESOLVED'
            else:
                record.update(status='OBSERVED', noninterference=noninterference,
                              pdk_frames=frames, pdk_label_components=nodes)
        key_results[key] = record
    generated = {'generate.py', 'toolchain.py', 'reference.spice'}
    for key in specs:
        generated.update((f'{key}/request.json', f'{key}/generate.tcl'))
    generated.update(f'{key}/trace/generate.tcl' for key in eligible)
    planned = plan.get('inputs_sha256')
    if not isinstance(planned, dict) or set(planned) != {str(root / name) for name in generated}:
        raise ValueError('role trace planned generated-input denominator differs')
    for relative in generated:
        observed = _artifact(root, relative, checkpoint)
        if observed['sha256'] != planned[str(root / relative)]:
            raise ValueError('role trace generated input differs from plan')
        if relative in file_identities and file_identities[relative] != observed:
            raise ValueError('role trace generated input changed during interpretation')
        file_identities[relative] = observed
    for relative, identity in file_identities.items():
        if _artifact(root, relative, checkpoint) != identity:
            raise ValueError('role trace artifact changed during observation')
    if _read_identity(runtime_file_identity, checkpoint) != runtime_bytes:
        raise ValueError('role trace source runtime changed during observation')
    occurrences = []
    for name, key in sorted(members.items()):
        for ordinal in range(specs[key]['occurrences']):
            occurrences.append({'source_instance': name, 'source_ordinal': ordinal, 'master_key': key})
    # Serialise once to detach any tuples returned by the trace observer.
    return json.loads(json.dumps({
        'schema': ('clay.primitive-role-trace/v1' if request['schema'].endswith('/v1')
                   else 'clay.primitive-role-trace/v2'), 'invocation_id': invocation_id,
        'request': request, 'source_identity': source_identity, 'runtime_sha256': runtime_sha256,
        'runtime_file': runtime_file_identity,
        'resource_plan': plan_identity, 'resource_evidence': evidence_identity,
        'runtime_observation': {'image': plan['image'], 'container_id': resource['container_id'],
            'platform': resource['actual_platform'], 'work_mount': '/work',
            'toolkit': toolkit, 'toolkit_authority': 'source-image runtime observation',
            'lyp': lyp_runtime, 'lyp_authority': 'source-image runtime observation'},
        'member_to_master': dict(sorted(members.items())), 'occurrences': occurrences,
        'occurrence_scope': 'source-declared ordinal only; no physical-side or net association',
        'masters': key_results, 'files': file_identities,
        'status': 'UNKNOWN', 'unknown': _UNKNOWN, 'physical_admission': False,
    }, sort_keys=True, allow_nan=False))


def verify_live_role_trace_manifest(*, output, request, invocation_id, source_identity,
                                    source_file_identity, runtime_sha256, runtime_file_identity, plan_identity,
                                    evidence_identity, checkpoint):
    """Reopen the stage's source and compare the complete recorded manifest."""
    source_bytes = _read_identity(source_file_identity, checkpoint)
    if len(source_bytes) > 16 * 1024 * 1024:
        raise ValueError('role trace source exceeds byte limit')
    specs, members = _source_specs(source_bytes, source_identity)
    expected = build_role_trace_manifest(output=output, specs=specs, members=members, request=request,
        invocation_id=invocation_id, source_identity=source_identity, runtime_sha256=runtime_sha256,
        runtime_file_identity=runtime_file_identity,
        plan_identity=plan_identity, evidence_identity=evidence_identity, checkpoint=checkpoint)
    path = Path(output).absolute() / _MANIFEST
    identity = _identity(path, checkpoint)
    actual = json.loads(_read_identity(identity, checkpoint))
    if actual != expected:
        raise ValueError('role trace manifest or source occurrence denominator differs')
    if _read_identity(source_file_identity, checkpoint) != source_bytes:
        raise ValueError('role trace source changed during replay')
    return identity


def replay_stage_role_trace(*, receipt_path, trusted_receipt_sha256, source_path, checkpoint,
                            managed_terminal_path=None, trusted_terminal_sha256=None):
    """Replay an opt-in capability from an independently trusted stage receipt.

    The caller obtains the receipt SHA256 from the task admission terminal,
    not from arbitrary input. This reader replays the source/member/spec
    denominator and retained observations; it does not execute Docker or
    treat retained cleanup as a fresh daemon observation.
    """
    from clay.flow.primitive_observation import read_stage_capture_records, verify_stage_capture_bytes
    receipt_path = Path(receipt_path).absolute()
    if (not re.fullmatch(r'[0-9a-f]{64}', str(trusted_receipt_sha256))
            or receipt_path.name != 'receipt.json' or receipt_path.parent.parent.name != 'attempts'):
        raise ValueError('invalid trusted primitive stage receipt identity')
    receipt_bytes = _read(receipt_path, checkpoint, limit=64 * 1024)
    if sha256(receipt_bytes).hexdigest() != trusted_receipt_sha256:
        raise ValueError('trusted primitive stage receipt changed')
    receipt = json.loads(receipt_bytes)
    if receipt_path.parent.name != receipt.get('invocation_id'):
        raise ValueError('role trace receipt invocation differs')
    records = read_stage_capture_records(receipt_path=receipt_path,
        trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=checkpoint,
        managed_terminal_path=managed_terminal_path, trusted_terminal_sha256=trusted_terminal_sha256)
    if len(records) != 1 or records[0].get('kind') != 'capture':
        raise ValueError('role trace requires the unique stage capture')
    record = records[0]['record']
    payload = verify_stage_capture_bytes(record, receipt_path=receipt_path,
        trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=checkpoint,
        managed_terminal_path=managed_terminal_path, trusted_terminal_sha256=trusted_terminal_sha256)
    start_identity = receipt.get('start')
    if not isinstance(start_identity, dict) or Path(start_identity.get('path', '')).parent != receipt_path.parent:
        raise ValueError('role trace stage start belongs to another attempt')
    start = json.loads(_read_identity(start_identity, checkpoint))
    if (receipt.get('schema') == 'clay.primitive-stage-receipt/v1'
            and start.get('schema') == 'clay.primitive-stage-start/v1'):
        if ('role_trace_request' in start or 'role_trace_request' in receipt
                or 'role_trace_manifest' in receipt or 'role_trace' in payload):
            raise ValueError('legacy stage cannot carry role trace evidence')
        return {'schema': 'clay.primitive-role-trace-replay/v1', 'status': 'NOT_REQUESTED',
                'physical_admission': False}
    if (receipt.get('schema') != 'clay.primitive-stage-receipt/v2'
            or start.get('schema') != 'clay.primitive-stage-start/v2'):
        raise ValueError('role trace stage version mismatch')
    request = _request(start.get('role_trace_request'))
    from clay.flow.run_admission import _checked_role_trace_request
    _checked_role_trace_request(request, start.get('budget'))
    if receipt.get('role_trace_request') != request:
        raise ValueError('role trace receipt request differs from start')
    manifest_identity = receipt.get('role_trace_manifest')
    if payload.get('role_trace') != {'request': request, 'manifest': manifest_identity}:
        raise ValueError('role trace capture does not bind the requested manifest')
    roots = payload.get('output_roots')
    if not isinstance(roots, list) or len(roots) != 1:
        raise ValueError('role trace capture output root differs')
    output = Path(roots[0]).absolute()
    if (not isinstance(manifest_identity, dict)
            or Path(manifest_identity.get('path', '')) != output / _MANIFEST):
        raise ValueError('role trace manifest belongs to another producer')
    output_entry = payload.get('outputs', {}).get(str(output / _MANIFEST))
    if output_entry != {key: manifest_identity[key] for key in ('sha256', 'size_bytes')}:
        raise ValueError('role trace manifest is not in the stage capture')
    bindings_identity = start.get('source_bindings')
    if (not isinstance(bindings_identity, dict)
            or Path(bindings_identity.get('path', '')).parent != receipt_path.parent):
        raise ValueError('role trace source bindings belong to another attempt')
    bindings = json.loads(_read_identity(bindings_identity, checkpoint))
    design = bindings.get('design')
    runtime = bindings.get('runtime')
    if (not isinstance(design, dict) or not isinstance(runtime, dict)
            or Path(design.get('path', '')) != Path(source_path).resolve(strict=True)):
        raise ValueError('role trace caller source differs from stage source')
    source_entry = payload.get('source_files', {}).get(design['path'])
    if source_entry != {key: design[key] for key in ('sha256', 'size_bytes')}:
        raise ValueError('role trace source is not in the trusted capture')
    _read_identity(design, checkpoint)
    runtime_entry = payload.get('source_files', {}).get(runtime['path'])
    if runtime_entry != {key: runtime[key] for key in ('sha256', 'size_bytes')}:
        raise ValueError('role trace runtime is not in the trusted capture')
    _read_identity(runtime, checkpoint)
    plan_identity, evidence_identity = receipt.get('resource_plan'), receipt.get('resource_evidence')
    actual_identity = verify_live_role_trace_manifest(output=output, request=request,
        invocation_id=receipt['invocation_id'], source_identity=payload['source_identity'],
        source_file_identity=design, runtime_sha256=runtime['sha256'], runtime_file_identity=runtime,
        plan_identity=plan_identity,
        evidence_identity=evidence_identity, checkpoint=checkpoint)
    if actual_identity != manifest_identity:
        raise ValueError('role trace manifest identity differs from stage receipt')
    manifest = json.loads(_read_identity(manifest_identity, checkpoint))
    if verify_stage_capture_bytes(record, receipt_path=receipt_path,
        trusted_receipt_sha256=trusted_receipt_sha256, checkpoint=checkpoint,
        managed_terminal_path=managed_terminal_path, trusted_terminal_sha256=trusted_terminal_sha256) != payload:
        raise ValueError('stage capture changed during role trace replay')
    return {'schema': 'clay.primitive-role-trace-replay/v1', 'status': 'UNKNOWN',
            'request': request, 'manifest': manifest_identity, 'observation': manifest,
            'physical_admission': False}
