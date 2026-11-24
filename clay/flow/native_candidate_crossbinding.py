"""Draft caller check using the existing comparator native evidence consumer.

This checks byte correspondence to the caller's claimed layout/source; source
to layout production remains the canonical flow's independent responsibility.
"""
from hashlib import sha256
import json
from pathlib import Path

from clay.flow.candidate_artifacts import CandidateArtifactChanged, source_identity, require_identity


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_native_candidate(*, root, source, source_before, gds, gds_before, pex, pex_before):
    from circuits.jku_comparator.runner import validate_native_pex

    root=Path(root).resolve()
    for path, expected in ((source,source_before),(gds,gds_before),(pex,pex_before)):
        require_identity(root,path,expected)
    sidecar=Path(str(pex)+'.native-verification.json')
    sidecar_before=source_identity(root,sidecar)
    observed=validate_native_pex(pex)
    summary=Path(observed['native_summary']['path'])
    manifest=Path(observed['native_manifest']['path'])
    summary_before=source_identity(root,summary)
    manifest_before=source_identity(root,manifest)
    data=json.loads(summary.read_text(encoding='utf-8'))
    if (observed['candidate_source']['sha256']!=source_before['sha256']
        or data['inputs']['reference_spice']['sha256']!=source_before['sha256']
        or data['inputs']['gds']['sha256']!=gds_before['sha256']
        or data['pex']['native_spice']['sha256']!=pex_before['sha256']
        or observed['native_summary']['sha256']!=summary_before['sha256']
        or observed['native_manifest']['sha256']!=manifest_before['sha256']):
        raise CandidateArtifactChanged('native evidence differs from actual candidate source, GDS or PEX')
    for path, expected in ((source,source_before),(gds,gds_before),(pex,pex_before),
                           (sidecar,sidecar_before),(summary,summary_before),(manifest,manifest_before)):
        require_identity(root,path,expected)
    return {'scope':'native evidence replay and same-candidate source, GDS and PEX bytes',
        'source':source_before,'gds':gds_before,'pex':pex_before,'sidecar':sidecar_before,
        'native_summary':summary_before,'native_manifest':manifest_before}
