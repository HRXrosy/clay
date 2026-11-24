"""Observe a caller-declared scope on the current local Docker endpoint.

This is a present-tense observation, not evidence of historical cleanup or of
resources on another daemon. The production interruption remains unaccounted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from clay.flow.owned_process import run_owned_process
from clay.flow.native_owned_container import _cli_identity, _mount as _mount_path


@dataclass(frozen=True)
class CurrentQuiescenceRequest:
    """Explicit opt-in to observe specific output roots on one local endpoint."""
    endpoint: str
    output_roots: tuple[Path, ...]

    def __post_init__(self):
        if not isinstance(self.endpoint, str) or not self.endpoint.lower().startswith('npipe:////./pipe/'):
            raise ValueError('current observation requires an explicit local Docker endpoint')
        if not isinstance(self.output_roots, (tuple, list)) or not self.output_roots:
            raise ValueError('current observation requires at least one output root')
        object.__setattr__(self, 'output_roots', tuple(Path(path) for path in self.output_roots))


def observe_current_quiescence(request, *, project_root, directory, checkpoint, remaining):
    """Scan all containers twice under the stage lock; never remove resources."""
    if type(request) is not CurrentQuiescenceRequest:
        raise TypeError('invalid current quiescence request')
    if os.name != 'nt':
        raise ValueError('current local Docker observation requires Windows')
    if os.environ.get('DOCKER_HOST') != request.endpoint or os.environ.get('DOCKER_CONTEXT'):
        raise ValueError('stage Docker environment must select the observed endpoint')
    root = Path(project_root).resolve(strict=True)
    resolved = tuple(path.resolve(strict=True) for path in request.output_roots)
    if any(not path.is_dir() or not path.is_relative_to(root) for path in resolved):
        raise ValueError('observation roots must be existing project directories')
    normalized = tuple(_mount_path(path) for path in resolved)
    if len(set(normalized)) != len(normalized):
        raise ValueError('duplicate current observation root')
    docker = shutil.which('docker')
    if docker is None:
        raise FileNotFoundError('docker')
    docker = str(Path(docker).resolve(strict=True))
    docker_identity = _cli_identity(docker, checkpoint=checkpoint)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    commands = []

    def command(*args):
        checkpoint()
        index = len(commands)
        stdout = directory / f'{index:03d}.stdout'
        stderr = directory / f'{index:03d}.stderr'
        argv = [docker, '-H', request.endpoint, *args]
        result = run_owned_process(argv, stdout_path=stdout, stderr_path=stderr,
                                   timeout_seconds=remaining(), cleanup_seconds=5,
                                   checkpoint=checkpoint, cwd=root, natural_exit_grace_seconds=.5)
        out, err = stdout.read_bytes(), stderr.read_bytes()
        commands.append({'argv': argv, 'result': asdict(result),
                         'stdout': {'path': str(stdout), 'sha256': sha256(out).hexdigest(), 'size_bytes': len(out)},
                         'stderr': {'path': str(stderr), 'sha256': sha256(err).hexdigest(), 'size_bytes': len(err)}})
        if result.exit_code != 0 or result.timed_out or result.descendants_terminated or result.job_active_after != 0:
            raise ValueError('current Docker observation command failed: ' + repr(asdict(result)))
        checkpoint()
        return out

    daemon_before = command('info', '--format', '{{.ID}}').decode('utf-8').strip()
    scans = []
    for _ in range(2):
        identifiers = command('ps', '-aq', '--no-trunc').decode('utf-8').splitlines()
        matches = []
        for identifier in identifiers:
            if len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
                raise ValueError('unexpected Docker container identity')
            mounts = json.loads(command('inspect', '--format', '{{json .Mounts}}', identifier))
            if not isinstance(mounts, list):
                raise ValueError('unexpected Docker mount inventory')
            for mount in mounts:
                if mount.get('Type') != 'bind':
                    continue
                source = _mount_path(mount.get('Source', ''))
                if any(source == item or source.startswith(item + '\\') or item.startswith(source + '\\')
                       for item in normalized):
                    matches.append(identifier)
                    break
        scans.append({'containers': len(identifiers), 'matching_containers': sorted(set(matches))})
    daemon_after = command('info', '--format', '{{.ID}}').decode('utf-8').strip()
    if (_cli_identity(docker, checkpoint=checkpoint) != docker_identity
            or not daemon_before or daemon_before != daemon_after
            or any(row['matching_containers'] for row in scans)):
        raise ValueError('current Docker endpoint is not quiescent for the requested roots')
    return {'schema': 'clay.current-quiescence-observation/v1', 'endpoint': request.endpoint,
            'docker_cli': docker_identity, 'daemon_id': daemon_before, 'output_roots': [str(path) for path in resolved],
            'scans': scans, 'commands': commands, 'historical_cleanup': 'UNKNOWN',
            'other_endpoints': 'UNKNOWN', 'host_processes': 'UNKNOWN',
            'scope_complete': 'UNKNOWN', 'physical_admission': False}
