"""Source-bound native composite row passed to the existing placement caller."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from weakref import WeakKeyDictionary

from clay.layout.native_sky130 import NativeGeometryBundle, native_source_identity
from clay.layout.place.row_program import verify_row_program_materialization
from clay.layout.place.row_source_projection import replay_row_source_projection


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True, eq=False, init=False)
class NativeCompositeRow:
    cell: object
    native_bundle: object
    tech: object
    projection: object
    program: object
    materialization: object
    geometry: object
    receipt_path: Path
    receipt_sha256: str
    dependencies: tuple

    def __init__(self, *args, **kwargs):
        raise TypeError('native composite row requires its producer')

    def require_current(self, *, cell=None, bundle=None):
        captured = _CAPTURED.get(self)
        identity = (tuple(id(getattr(self, name)) for name in _IDENTITY_FIELDS),
                    tuple(id(row.geometry) for row in self.materialization.realizations))
        if captured is None or captured != identity:
            raise ValueError('native composite producer identity changed')
        if type(self.native_bundle) is not NativeGeometryBundle or (bundle is not None and bundle is not self.native_bundle):
            raise ValueError('native composite source bundle differs')
        self.native_bundle.require_current(self.cell)
        if cell is not None and cell is not self.cell:
            self.native_bundle.require_current(cell)
            if native_source_identity(cell) != native_source_identity(self.cell):
                raise ValueError('native composite source differs')
        for path, expected in self.dependencies:
            if _digest(path) != expected:
                raise ValueError('native composite producer dependency changed: '+str(path))
        if _digest(self.receipt_path) != self.receipt_sha256:
            raise ValueError('native composite producer receipt changed')
        receipt = json.loads(self.receipt_path.read_text(encoding='utf-8'))
        for name, expected in receipt['outputs'].items():
            if _digest(self.receipt_path.parent / name) != expected:
                raise ValueError('native composite producer output changed: '+name)
        for name, expected in receipt['inputs'].items():
            if _digest(Path(__file__).resolve().parents[2] / name) != expected:
                raise ValueError('native composite producer input changed: '+name)
        from clay.layout.place.row_replay_session import (
            _CURRENT as _REPLAY_CURRENT,
            current_row_source_program,
            require_current_row_materialization,
        )
        program = current_row_source_program(
            self.projection,
            cell or self.cell,
            self.tech,
            program=self.program,
            materialization=self.materialization,
        )
        if (program.members != self.program.members or
                program.to_canonical_bytes() != self.program.to_canonical_bytes()):
            raise ValueError('native composite source program changed')
        live_proof = _REPLAY_CURRENT.get()
        if (
            live_proof is None
            or live_proof.materialization is None
            or live_proof.replayed_source_program is None
        ):
            require_current_row_materialization(self.materialization, program=program, tech_id=self.tech.pdk_id)
        else:
            if (
                self.materialization is not live_proof.materialization
                or self.tech.pdk_id != live_proof.tech.pdk_id
                or not (
                    program is live_proof.program
                    or program is live_proof.replayed_source_program
                )
            ):
                require_current_row_materialization(self.materialization, program=program, tech_id=self.tech.pdk_id)
        realizations = {row.owner: row for row in self.materialization.realizations}
        entries = {row.owner: row for row in self.program.entries}
        if (not entries or len(entries) != len(self.program.entries)
                or len(realizations) != len(self.materialization.realizations)
                or set(realizations) != set(entries)
                or self.materialization.realizations[0].geometry is not self.geometry):
            raise ValueError('native composite materialization identity changed')
        witness = getattr(self.projection, 'native_witness', None)
        if witness is not None:
            sources = {name: geometry for name, geometry, _folder in witness.rows}
            if set(sources) != {entry.group for entry in entries.values()}:
                raise ValueError('native composite PCell source partition differs')
            for owner, realization in realizations.items():
                source = sources[entries[owner].group]
                actual = realization.geometry
                if (actual.joint_row_binding is not source.joint_row_binding
                        or actual.polygons is not source.polygons or actual.fingers is not source.fingers
                        or actual.channel_inventory is not source.channel_inventory):
                    raise ValueError('native composite geometry differs from produced PCell')
        # Every consumed occurrence must carry its real, source-bound body
        # component. Missing body contacts must not disappear with the row's
        # standalone native pin extraction.
        from clay.layout.native_sky130 import native_connect_model
        members = {row.device: row for row in self.program.members}
        seen = set()
        for owner, realization in realizations.items():
            entry = entries[owner]
            if seen.intersection(entry.consumed_devices):
                raise ValueError('native composite source partition overlaps')
            seen.update(entry.consumed_devices)
            binding = getattr(realization.geometry, 'joint_row_binding', None)
            if binding is None or binding.connect_model != native_connect_model('mos'):
                raise ValueError('native composite body connectivity model differs')
            body_sites = {(site.occurrence_id, site.owner, site.source_net)
                          for site in binding.expected_sites if site.allowed_roles == ('b',)}
            expected_bodies = {(occ.occurrence_id, occ.owner, members[occ.owner].body_net)
                               for occ in binding.occurrences}
            if not expected_bodies or body_sites != expected_bodies:
                raise ValueError('native composite source body access is incomplete')
        return self


_IDENTITY_FIELDS = ('cell','native_bundle','tech','projection','program','materialization','geometry',
                    'receipt_path','receipt_sha256','dependencies')
_CAPTURED = WeakKeyDictionary()


def _capture_native_composite_row(**fields):
    if set(fields) != set(_IDENTITY_FIELDS):
        raise ValueError('native composite producer fields differ')
    result = object.__new__(NativeCompositeRow)
    for name in _IDENTITY_FIELDS:
        object.__setattr__(result, name, fields[name])
    _CAPTURED[result] = (tuple(id(getattr(result, name)) for name in _IDENTITY_FIELDS),
                         tuple(id(row.geometry) for row in result.materialization.realizations))
    return result.require_current()
