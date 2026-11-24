"""Immutable source authority for one source-declared sizing invocation."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path


class SizingSourceChanged(ValueError):
    """A sizing source changed or disappeared during its evaluation."""


def spec_fingerprint(spec):
    """Freeze the complete spec, including nested mutable policy data."""
    from clay.flow.spec import CircuitSpec
    if not isinstance(spec, CircuitSpec):
        raise TypeError('source-bound sizing requires a CircuitSpec')
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError('unsupported value in sizing spec: ' + type(value).__name__)
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False, default=encode)
    return sha256(payload.encode('utf-8')).hexdigest()


def verify_spec_fingerprint(spec, expected):
    try:
        actual = spec_fingerprint(spec)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        # The spec was serializable when bound. A now-unserializable nested
        # value is identity drift, not a failed optimization sample.
        raise SizingSourceChanged('sizing circuit spec changed') from exc
    if actual != expected:
        raise SizingSourceChanged('sizing circuit spec changed')


@dataclass(frozen=True)
class SizingSource:
    path: Path
    sha256: str
    label: str = 'sizing authoring source'

    @classmethod
    def capture(cls, path):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError('sizing source is not a file')
        return cls(path, sha256(path.read_bytes()).hexdigest())

    def verify(self):
        try:
            actual = sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise SizingSourceChanged(self.label + ' is unavailable') from exc
        if actual != self.sha256:
            raise SizingSourceChanged(self.label + ' bytes changed')
        return self.path


def sizing_source(spec, authority=None):
    """Use a bound authority, or the historical case source when it exists."""
    if authority is not None:
        if type(authority) is not SizingSource:
            raise TypeError('sizing source must be a frozen SizingSource')
        authority.verify()
        return authority
    path = Path(spec.case_dir) / 'source' / 'design.clay'
    return SizingSource.capture(path) if path.is_file() else None


def require_source_matches_spec(spec, authority, *, full_projection=False):
    """Check a source; standalone sizing requires a complete spec projection."""
    from clay.flow.spec import _optimization_from_sizing, compile_flow_spec
    from clay.parser import parse_clay_source

    # The existing sizing candidate/promotion materializers use the same
    # parse_clay_source boundary and explicitly reject file-path imports.
    # Reject here before PRE rather than omit imported source identities.
    parsed = parse_clay_source(authority.verify().read_text(encoding='utf-8-sig'))
    cells = [cell for cell in parsed.cell_defs if cell.local_name == spec.top_cell]
    if (parsed.pdk_id != spec.pdk or len(cells) != 1
            or _optimization_from_sizing(cells[0]) != spec.optimization):
        raise ValueError('sizing source PDK, top or optimization differs from compiled spec')
    if full_projection:
        # CircuitSpec is a flow/quality/policy projection, not a full physical
        # source identity. The selected authority is independently frozen and
        # remains the sole source of candidate physical intent.
        projected = compile_flow_spec(cells[0], case_dir=spec.case_dir,
            verify_jobs=getattr(parsed, 'verify_jobs', None),
            policies=getattr(parsed, 'policies', None), discover_layout=False)
        if projected != spec:
            # A caller may have used the public default discovery mode, which
            # resolves an existing layout module's callable names. Permit
            # that exact projection as well; never ignore policy/KPI fields.
            discovered = compile_flow_spec(cells[0], case_dir=spec.case_dir,
                verify_jobs=getattr(parsed, 'verify_jobs', None),
                policies=getattr(parsed, 'policies', None), discover_layout=True)
            if discovered != spec:
                raise ValueError('sizing source does not reproduce the complete circuit spec')
    authority.verify()
