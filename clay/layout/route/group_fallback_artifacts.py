"""Partial group-route evidence: raw observations and validated complete pairs."""
from hashlib import sha256
import json
from pathlib import Path
import re

_HASH = re.compile(r'^[0-9a-f]{64}$')

def observations(work_dir):
    """Enumerate files without treating a partial or interrupted write as PASS."""
    root = Path(work_dir) / 'group-fallback'
    if not root.is_dir():
        return {}
    paths = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        paths.extend(p for p in sorted(folder.iterdir()) if p.is_file() and not p.is_symlink()
                     and (p.name == 'branch-events.jsonl'
                          or p.name.startswith('accepted-group-ledger-') and p.suffix == '.json'
                          or p.name.startswith('accepted-group-prefix-') and p.suffix == '.gds'))
    return {f'group_fallback_observation_{i:04d}': p for i, p in enumerate(paths)}

def completed_pairs(work_dir):
    """Validate hashes and bind each partial GDS to its accepted scene ledger."""
    raw = observations(work_dir)
    outputs = {}
    by_path = {str(path.resolve()): path for path in raw.values()}
    for path in raw.values():
        if not path.name.startswith('accepted-group-ledger-'):
            continue
        encoded = path.read_bytes()
        digest = sha256(encoded).hexdigest()
        if path.name != 'accepted-group-ledger-'+digest+'.json':
            raise ValueError('group fallback accepted ledger digest mismatch')
        ledger = json.loads(encoded)
        feedback = ledger.get('feedback', {})
        if not feedback.get('accepted'):
            raise ValueError('group fallback ledger is not accepted feedback')
        manifest = ledger.get('router_via_manifest')
        if (ledger.get('schema') != 'clay.group-fallback-placed-scene.v1'
            or feedback.get('router_via_status') != 'pass'
            or type(manifest) is not list
            or any(type(site) is not dict or type(site.get('cuts')) is not list for site in manifest)
            or feedback.get('router_cuts_checked') != sum(len(site['cuts']) for site in manifest)):
            raise ValueError('group fallback ledger lacks qualified router-cut evidence')
        expected = ledger.get('actual_gds_sha256')
        if type(expected) is not str or not _HASH.fullmatch(expected):
            raise ValueError('group fallback GDS digest is invalid')
        if ledger.get('actual_gds_name') != 'accepted-group-prefix-'+expected+'.gds':
            raise ValueError('group fallback GDS name does not bind digest')
        gds = path.parent / ledger['actual_gds_name']
        if str(gds.resolve()) not in by_path or sha256(gds.read_bytes()).hexdigest() != expected:
            raise ValueError('group fallback GDS missing or digest mismatch')
        outputs['group_fallback_ledger_'+digest] = path
        outputs['group_fallback_gds_'+expected] = gds
    for name, path in raw.items():
        if path.name == 'branch-events.jsonl':
            outputs[name] = path
    return outputs

