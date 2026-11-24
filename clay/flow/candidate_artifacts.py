"""Bind a sizing evaluation to its own produced source and extracted bytes.

This seal starts after the nominal POST flow returns. Native comparator PEX
publication and simulation still require their existing native sidecar replay;
this module does not reinterpret PEX or replace any signoff check.
"""
import json
from pathlib import Path

from clay.flow.run_workspace import artifact_identity
from clay.flow.sizing_source import SizingSourceChanged


class CandidateArtifactChanged(SizingSourceChanged):
    """An evaluated candidate's source, PEX, or journal binding changed."""


def _identity(root, path):
    try:
        # Pass the unresolved absolute path so the existing owned-path gate
        # can reject symlink/junction aliases before it resolves the file.
        return artifact_identity(Path(root).resolve(), Path(path).absolute())
    except (OSError, ValueError, TypeError) as exc:
        raise CandidateArtifactChanged('candidate artifact unavailable or outside its evaluation') from exc


def source_identity(root, path):
    return _identity(root, path)


def require_identity(root, path, expected):
    if _identity(root, path) != expected:
        raise CandidateArtifactChanged('candidate artifact bytes or identity changed')


def seal_candidate(*, root, evaluation_index, sizing, source, source_before, pex,
                   staged=None, staged_before=None, native_observation=None):
    root = Path(root).resolve()
    require_identity(root, source, source_before)
    if staged is not None:
        require_identity(root, staged, staged_before)
    value = {'schema': 'clay.sizing-candidate-artifacts/v1', 'evaluation_index': int(evaluation_index),
             'evaluation_root': str(root), 'canonical_sizing': dict(sizing),
             'source': source_before, 'pex': _identity(root, pex),
             'staged_clay': staged_before, 'native_observation': native_observation}
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def verify_candidate(encoded, record, canonical_sizing):
    if not isinstance(encoded, str):
        raise CandidateArtifactChanged('candidate has no retained evaluation identity')
    value = json.loads(encoded)
    if (record.get('candidate_artifact_seal') != value
        or record.get('evaluation_index') != value['evaluation_index']
        or dict(canonical_sizing) != value['canonical_sizing']):
        raise CandidateArtifactChanged('candidate journal does not match its retained evaluation identity')
    root = Path(value['evaluation_root'])
    for kind, field in (('source', 'candidate_source'), ('pex', 'candidate_pex'),
                        ('staged_clay', 'candidate_design_clay')):
        identity = value[kind]
        if identity is None:
            if record.get(field) is not None:
                raise CandidateArtifactChanged('candidate journal invents an absent artifact')
            continue
        if kind == 'staged_clay' and record.get('candidate_design_clay_sha256') != identity['sha256']:
            raise CandidateArtifactChanged('candidate staged Clay digest differs from retained identity')
        path = root / identity['path']
        try:
            same = Path(str(record.get(field) or '')).absolute() == path.absolute()
        except (OSError, ValueError) as exc:
            raise CandidateArtifactChanged('candidate journal path is invalid') from exc
        if not same:
            raise CandidateArtifactChanged('candidate journal points to another evaluation artifact')
        require_identity(root, path, identity)
    native = value.get('native_observation')
    if native is not None:
        from clay.flow.native_candidate_crossbinding import verify_native_candidate
        from clay.flow.execution_budget import ExecutionStopped

        def verify_retained_native_files():
            for kind in ('source', 'gds', 'pex', 'sidecar', 'native_summary', 'native_manifest'):
                identity = native[kind]
                require_identity(root, root / identity['path'], identity)

        # Compare the original nominal-POST proof before replay. Otherwise a
        # changed sidecar can fail inside the native reader as an ordinary
        # candidate-level ValueError before its prior identity is checked.
        verify_retained_native_files()
        try:
            current = verify_native_candidate(root=root, source=root / value['source']['path'],
                source_before=value['source'], gds=root / native['gds']['path'], gds_before=native['gds'],
                pex=root / value['pex']['path'], pex_before=value['pex'])
        except ExecutionStopped:
            raise
        except Exception as original:
            try:
                verify_retained_native_files()
            except CandidateArtifactChanged as changed:
                raise changed from original
            raise
        verify_retained_native_files()
        if current != native:
            raise CandidateArtifactChanged('native candidate evidence changed since nominal POST')
    return value


def capture_accepted_source(record):
    """Freeze the exact source identity already validated by acceptance."""
    try:
        seal = record['candidate_artifact_seal']
        if not isinstance(seal, dict) or seal.get('schema') != 'clay.sizing-candidate-artifacts/v1':
            raise ValueError('missing accepted candidate seal')
        if record.get('evaluation_index') != seal['evaluation_index']:
            raise ValueError('accepted candidate index differs')
        root = Path(seal['evaluation_root'])
        identity = dict(seal['source'])
        path = root / identity['path']
        if not root.is_absolute() or Path(str(record.get('candidate_source') or '')).absolute() != path.absolute():
            raise ValueError('accepted candidate source path differs')
        if set(identity) != {'path', 'sha256', 'size_bytes'}:
            raise ValueError('accepted candidate source identity differs')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise CandidateArtifactChanged('accepted candidate source evidence unavailable') from exc
    require_identity(root, path, identity)
    return root, path, identity


def verify_promoted_source(accepted, output_root, promoted):
    """Require final promoted bytes to equal the accepted sealed source."""
    root, path, identity = accepted
    require_identity(root, path, identity)
    actual = source_identity(Path(output_root).resolve(), promoted)
    if (actual['sha256'], actual['size_bytes']) != (identity['sha256'], identity['size_bytes']):
        raise CandidateArtifactChanged('promoted schematic differs from accepted candidate source')
    return actual
