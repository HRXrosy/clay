"""Frozen placement data for the router's exact internal-edge rail predicate.

These records carry no source or placement authority. The native compiler must
capture their complete original stamp inventory and retain the resulting digest.
Conductor boxes are master-local and may extend beyond their stamp's envelope.
Only actual stamp envelopes define the outline; conductors never enlarge it.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, fields, replace
import hashlib
from itertools import combinations
import json
import marshal
from operator import attrgetter
from collections.abc import Mapping

from clay.layout.route.emit_gds import _nearest_on_grid
from clay.layout.route.rail_plan import (
    MAX_COORD_ABS_NM,
    _rects_touch_or_overlap,
    _rects_within_spacing,
)


EDGE_RAIL_PREDICATE_ID = "clay.edge-rail-reservation.physical-union-nearest-grid.v1"
EDGE_RAIL_ACCESS_PREDICATE_ID = "clay.edge-rail-access.closed-ring-via-domain.v1"
NATIVE_RAIL_ACCESS_PAIR_PREDICATE_ID = "clay.native-rail-access-pair-separation.v1"
EDGE_RAIL_WIDE_PREDICATE_ID = "clay.edge-rail-wide-rectangle-pdk.v1"
_LIMIT = MAX_COORD_ABS_NM
_VALIDATED_SNAPSHOTS = OrderedDict()
_SNAPSHOT_CACHE_LIMIT = 8
_RECORD_FIELDS = {}


def _record_snapshot(rows, row_type):
    # Exact types are checked before serialization: bool and int, as well as
    # tuples and lists, must never share a successful validation cache entry.
    _require(type(rows) is tuple and all(type(row) is row_type for row in rows),
             "snapshot inventory has foreign or mutable records")
    names, getter = _RECORD_FIELDS.setdefault(row_type, (
        tuple(field.name for field in fields(row_type)),
        attrgetter(*(field.name for field in fields(row_type)))))
    return names, tuple(getter(row) for row in rows)


def _require(condition, message):
    if not condition:
        raise ValueError(f"edge rail reservation {message}")


def _text(value, label):
    _require(type(value) is str and 0 < len(value) <= 1024, f"{label} must be a nonempty string")


def _sha(value, label):
    _require(type(value) is str and len(value) == 64
             and all(char in "0123456789abcdef" for char in value), f"{label} must be SHA256")


def _integer(value, label, *, minimum=0, maximum=2 * _LIMIT):
    _require(type(value) is int and minimum <= value <= maximum,
             f"{label} must be a bounded exact integer")


def _box(value, label):
    _require(type(value) is tuple and len(value) == 4
             and all(type(item) is int and abs(item) <= _LIMIT for item in value)
             and value[0] < value[2] and value[1] < value[3],
             f"{label} must be a bounded positive-area integer bbox")


def _layer(value):
    _require(type(value) is tuple and len(value) == 2
             and all(type(item) is int and 0 <= item <= 65535 for item in value),
             "layer must be an exact GDS layer/datatype pair")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, slots=True)
class EdgeRailDemandIR:
    demand_id: str
    declaration_ref: str
    declaration_sha256: str
    net: str
    layer_n: int
    layer_key: tuple[int, int]
    actual_axis: str
    edge: str
    width_nm: int
    min_space_nm: int
    terminal_count: int
    joint_component_ids: tuple[str, ...] = ()
    access_layer_keys: tuple[tuple[int, int], ...] = ()
    access_clearance_nm: int = 0
    offset_nm: int = 0
    # A guaranteed wide rectangular rail has a stronger PDK clearance than the
    # ordinary same-layer minimum.  Zero retains the historical base-only IR.
    wide_space_nm: int = 0

    def __post_init__(self):
        for field in ("demand_id", "declaration_ref", "net"):
            _text(getattr(self, field), field)
        _sha(self.declaration_sha256, "declaration identity")
        _integer(self.layer_n, "metal layer", minimum=1, maximum=32)
        _layer(self.layer_key)
        _require(self.actual_axis in ("H", "V") and type(self.actual_axis) is str,
                 "actual axis must be H or V")
        _require(type(self.edge) is str and self.edge in (
            ("top", "bottom") if self.actual_axis == "H" else ("left", "right")),
            "edge is incompatible with actual axis")
        _integer(self.width_nm, "width", minimum=1)
        _integer(self.offset_nm, "inward offset")
        _integer(self.min_space_nm, "spacing")
        _integer(self.wide_space_nm, "wide-rail spacing")
        _require(not self.wide_space_nm or self.wide_space_nm > self.min_space_nm,
                 "wide-rail spacing must strengthen the base spacing")
        _integer(self.terminal_count, "terminal count")
        _require(type(self.joint_component_ids) is tuple and len(self.joint_component_ids) <= 262144,
                 'joint component inventory must be bounded')
        for identity in self.joint_component_ids:
            _text(identity, 'joint component identity')
        _require(self.joint_component_ids == tuple(sorted(set(self.joint_component_ids)))
                 and len(self.joint_component_ids) <= self.terminal_count, 'joint AND demand denominator differs')
        _require(type(self.access_layer_keys) is tuple, 'access layer inventory must be immutable')
        for key in self.access_layer_keys:
            _layer(key)
        _require(self.access_layer_keys == tuple(sorted(set(self.access_layer_keys)))
                 and self.layer_key not in self.access_layer_keys
                 and len(self.access_layer_keys) < self.layer_n,
                 'access layer inventory is duplicated or includes the rail layer')
        _integer(self.access_clearance_nm, 'access clearance')
        _require(bool(self.access_layer_keys) == bool(self.access_clearance_nm),
                 'access layers and clearance must be present together')

    def payload(self):
        result = asdict(self)
        if not self.wide_space_nm:
            result.pop('wide_space_nm')
        if not self.offset_nm:
            result.pop('offset_nm')
        if not self.joint_component_ids:
            result.pop('joint_component_ids')
        if not self.access_layer_keys:
            result.pop('access_layer_keys')
            result.pop('access_clearance_nm')
        return result


@dataclass(frozen=True, slots=True)
class EdgeRailStampIR:
    """Actual raw stamp extent; offset is its lower left in the master frame."""

    stamp_id: str
    master_id: str
    offset_nm: tuple[int, int]
    bbox_nm: tuple[int, int, int, int]
    geometry_sha256: str
    access_sha256: str
    conductor_atom_count: int
    guard_source_members: tuple[str, ...] = ()
    guard_declaration_sha256: str = ""

    def __post_init__(self):
        _text(self.stamp_id, "stamp id")
        _text(self.master_id, "master id")
        _box(self.bbox_nm, "stamp bbox")
        _require(type(self.offset_nm) is tuple and len(self.offset_nm) == 2
                 and all(type(item) is int for item in self.offset_nm)
                 and self.offset_nm == self.bbox_nm[:2], "stamp offset must equal its bbox lower left")
        _sha(self.geometry_sha256, "stamp geometry identity")
        _sha(self.access_sha256, "stamp access identity")
        _integer(self.conductor_atom_count, "stamp conductor atom count")
        _require(type(self.guard_source_members) is tuple
                 and self.guard_source_members == tuple(sorted(set(self.guard_source_members)))
                 and all(type(name) is str and name for name in self.guard_source_members),
                 "guard source members must be exact canonical names")
        _require(bool(self.guard_source_members) == bool(self.guard_declaration_sha256),
                 "guard source members and declaration identity must occur together")
        if self.guard_source_members:
            _sha(self.guard_declaration_sha256, "guard declaration identity")


@dataclass(frozen=True, slots=True)
class EdgeRailConductorAtom:
    stamp_id: str
    master_id: str
    layer_key: tuple[int, int]
    net: str | None
    bbox_nm: tuple[int, int, int, int]
    kind: str

    def __post_init__(self):
        _text(self.stamp_id, "atom stamp id")
        _text(self.master_id, "atom master id")
        _layer(self.layer_key)
        if self.net is not None:
            _text(self.net, "atom net")
        _box(self.bbox_nm, "atom bbox")
        _text(self.kind, "atom kind")


@dataclass(frozen=True, slots=True)
class EdgeRailReservationIR:
    """Validated frozen data; a constructor or digest grants no live authority.

    Source and topology digests name the producer's canonical parsed source AST
    and complete bridged topology data, respectively, rather than file bytes.
    Every bbox uses the original rigid master's integer-nanometre origin.
    """

    source_sha256: str
    topology_sha256: str
    pdk_id: str
    pdk_provenance_sha256: str
    manufacturing_grid_nm: int
    masters: tuple[tuple[str, int, int], ...]
    demands: tuple[EdgeRailDemandIR, ...]
    stamps: tuple[EdgeRailStampIR, ...]
    atoms: tuple[EdgeRailConductorAtom, ...]
    predicate_id: str = EDGE_RAIL_PREDICATE_ID
    native_access_pair_clearances_nm: tuple[tuple[str, str, int], ...] = ()
    native_feol_pairs: tuple = ()
    right_margin_nm: int = 0
    top_margin_nm: int = 0

    def __post_init__(self):
        self._validated_data()

    def _validated_data(self):
        from .native_feol_reservation import NativeFeolPairIR, master_displacement_covers
        feol_names, feol_pairs = _record_snapshot(getattr(self, 'native_feol_pairs', ()), NativeFeolPairIR)
        demand_names, demands = _record_snapshot(self.demands, EdgeRailDemandIR)
        stamp_names, stamps = _record_snapshot(self.stamps, EdgeRailStampIR)
        atom_names, atoms = _record_snapshot(self.atoms, EdgeRailConductorAtom)
        state = (self.source_sha256, self.topology_sha256, self.pdk_id,
                 self.pdk_provenance_sha256, self.manufacturing_grid_nm,
                 self.masters, demands, stamps, atoms, self.predicate_id,
                 getattr(self, 'native_access_pair_clearances_nm', ()), feol_pairs,
                 getattr(self, 'right_margin_nm', 0),
                 getattr(self, 'top_margin_nm', 0))
        try:
            # marshal is a private in-memory key, never an artifact format or
            # untrusted input. Its typed encoding distinguishes True from 1 and
            # catches nested mutations without Python deep-copy/asdict loops.
            key = marshal.dumps(state, 4)
        except (ValueError, TypeError):
            self._validate()
            raise ValueError("edge rail reservation has non-primitive snapshot data")
        cached = _VALIDATED_SNAPSHOTS.get(key)
        if cached is not None:
            _VALIDATED_SNAPSHOTS.move_to_end(key)
            return cached
        self._validate()
        demand_rows = [dict(zip(demand_names, row)) for row in demands]
        for row in demand_rows:
            if not row['wide_space_nm']:
                row.pop('wide_space_nm')
            if not row['joint_component_ids']:
                row.pop('joint_component_ids')
            if not row['access_layer_keys']:
                row.pop('access_layer_keys')
                row.pop('access_clearance_nm')
        stamp_rows = [dict(zip(stamp_names, row)) for row in stamps]
        for row in stamp_rows:
            if not row['guard_source_members']:
                row.pop('guard_source_members')
                row.pop('guard_declaration_sha256')
        atom_rows = [dict(zip(atom_names, row)) for row in atoms]
        cached = {
            'masters': tuple(sorted(self.masters)),
            'demands': tuple(sorted(demand_rows, key=lambda row: row['demand_id'])),
            'stamps': tuple(sorted(stamp_rows, key=lambda row: row['stamp_id'])),
            'atoms': tuple(sorted(atom_rows, key=_json)),
            'dimensions': {name: (width, height) for name, width, height in self.masters},
            'native_access_pairs': getattr(self, 'native_access_pair_clearances_nm', ()),
            'native_feol_pairs': tuple(dict(zip(feol_names, row)) for row in feol_pairs),
            'native_feol_covers': master_displacement_covers(getattr(self, 'native_feol_pairs', ())),
            'domains': {row.demand_id: _lower_access_domains(self.atoms, row)
                        for row in self.demands},
            'sha256': None,
        }
        # The entry contains primitive immutable field values, never the live
        # dataclass records. A forged object.__setattr__ cannot poison an older
        # entry subsequently reached by another equivalent, valid reservation.
        _VALIDATED_SNAPSHOTS[key] = cached
        while len(_VALIDATED_SNAPSHOTS) > _SNAPSHOT_CACHE_LIMIT:
            _VALIDATED_SNAPSHOTS.popitem(last=False)
        return cached

    def _validate(self):
        for name in ("source_sha256", "topology_sha256", "pdk_provenance_sha256"):
            _sha(getattr(self, name), name)
        _text(self.pdk_id, "PDK id")
        _integer(self.manufacturing_grid_nm, "manufacturing grid", minimum=1, maximum=_LIMIT)
        right_margin_nm = getattr(self, 'right_margin_nm', 0)
        _integer(right_margin_nm, "right margin", minimum=0, maximum=_LIMIT)
        _require(right_margin_nm % self.manufacturing_grid_nm == 0, "right margin is off manufacturing grid")
        top_margin_nm = getattr(self, 'top_margin_nm', 0)
        _integer(top_margin_nm, "top margin", minimum=0, maximum=_LIMIT)
        _require(top_margin_nm % self.manufacturing_grid_nm == 0, "top margin is off manufacturing grid")
        _require(self.predicate_id == EDGE_RAIL_PREDICATE_ID, "predicate identity is unsupported")
        _require(type(self.masters) is tuple and bool(self.masters), "master inventory is empty or mutable")
        for row in self.masters:
            _require(type(row) is tuple and len(row) == 3, "master dimension inventory is malformed")
            _text(row[0], "master id")
            _integer(row[1], "master width", minimum=1)
            _integer(row[2], "master height", minimum=1)
        owners = {row[0] for row in self.masters}
        _require(len(owners) == len(self.masters), "master inventory has duplicates")
        native_pairs = getattr(self, 'native_access_pair_clearances_nm', ())
        _require(type(native_pairs) is tuple and len(native_pairs) <= 262144,
                 "native access pair inventory must be bounded and immutable")
        for row in native_pairs:
            _require(type(row) is tuple and len(row) == 3,
                     "native access pair inventory is malformed")
            _text(row[0], "native access left owner")
            _text(row[1], "native access right owner")
            _require(row[0] in owners and row[1] in owners and row[0] < row[1],
                     "native access pair must name distinct canonical original masters")
            _integer(row[2], "native access pair clearance", minimum=1)
        _require(native_pairs == tuple(sorted(native_pairs))
                 and len({row[:2] for row in native_pairs}) == len(native_pairs),
                 "native access pair inventory has duplicates or is not canonical")
        for name, row_type in (("demands", EdgeRailDemandIR), ("stamps", EdgeRailStampIR),
                               ("atoms", EdgeRailConductorAtom)):
            rows = getattr(self, name)
            _require(type(rows) is tuple and all(type(row) is row_type for row in rows),
                     f"{name} inventory has foreign or mutable records")
            for row in rows:
                row.__post_init__()
        _require(bool(self.demands) or bool(getattr(self, 'native_feol_pairs', ())), "demand inventory is empty")
        _require(len({row.demand_id for row in self.demands}) == len(self.demands)
                 and len({row.declaration_ref for row in self.demands}) == len(self.demands),
                 "demand inventory has duplicate identities")
        _require(all(row.width_nm % self.manufacturing_grid_nm == 0 for row in self.demands),
                 "demand width is off manufacturing grid")
        _require(all(row.offset_nm % self.manufacturing_grid_nm == 0 for row in self.demands),
                 "demand offset is off manufacturing grid")
        layer_numbers, layer_keys = {}, {}
        for row in self.demands:
            _require(layer_numbers.setdefault(row.layer_n, row.layer_key) == row.layer_key
                     and layer_keys.setdefault(row.layer_key, row.layer_n) == row.layer_n,
                     "demand layer inventory is inconsistent")
        by_stamp = {row.stamp_id: row for row in self.stamps}
        _require(bool(self.stamps) and len(by_stamp) == len(self.stamps)
                 and {row.master_id for row in self.stamps} == owners,
                 "stamp inventory must cover every master exactly by named stamps")
        for atom in self.atoms:
            stamp = by_stamp.get(atom.stamp_id)
            _require(stamp is not None and stamp.master_id == atom.master_id,
                     "atom inventory references a foreign stamp or master")
        counts = Counter(row.stamp_id for row in self.atoms)
        _require(all(counts[row.stamp_id] == row.conductor_atom_count for row in self.stamps),
                 "stamp conductor atom inventory count differs")
        from .native_feol_reservation import NativeFeolPairIR
        feol_pairs = getattr(self, 'native_feol_pairs', ())
        _require(type(feol_pairs) is tuple and len(feol_pairs) <= 262144,
                 "native FEOL inventory must be bounded and immutable")
        identities = []
        for row in feol_pairs:
            _require(type(row) is NativeFeolPairIR, "native FEOL record has a foreign type")
            row.__post_init__()
            identities.append((row.left_stamp, row.right_stamp, row.rule_id, row.left_kind, row.right_kind))
            for side in ('left', 'right'):
                stamp = by_stamp.get(getattr(row, side + '_stamp'))
                _require(stamp is not None and stamp.master_id == getattr(row, side + '_master'),
                         "native FEOL domain references a foreign original stamp")
                for box in getattr(row, side + '_rectangles_nm'):
                    _require(stamp.bbox_nm[0] <= box[0] < box[2] <= stamp.bbox_nm[2]
                             and stamp.bbox_nm[1] <= box[1] < box[3] <= stamp.bbox_nm[3],
                             "native FEOL local domain escapes its actual source stamp")
        _require(identities == sorted(set(identities)), "native FEOL inventory is not canonical")

    def with_native_access_pair_clearances(self, mapping):
        """Bind source-derived future access, separately from emitted geometry.

        Current layer-disjoint footprints cannot discharge an obligation for a
        future via landing. The exact master pair clause therefore travels in
        the existing closed rail IR through CP, hard replay and certificates.
        A nonempty bound obligation cannot be silently replaced or removed.
        """
        self._validated_data()
        _require(isinstance(mapping, Mapping), "native access pairs must be a mapping")
        _require(len(mapping) <= 262144, "native access pair mapping is too large")
        owners = {row[0] for row in self.masters}
        rows, seen = [], set()
        for pair, gap in mapping.items():
            _require(type(pair) is tuple and len(pair) == 2,
                     "native access pair key must be an exact owner tuple")
            for owner in pair:
                _text(owner, "native access owner")
                _require(owner in owners, "native access pair references a foreign master")
            _require(pair[0] != pair[1], "native access pair references the same master")
            canonical = tuple(sorted(pair))
            _require(canonical not in seen, "native access pair has duplicate orientations")
            _integer(gap, "native access pair clearance", minimum=1)
            seen.add(canonical)
            rows.append((*canonical, gap))
        frozen = tuple(sorted(rows))
        current = getattr(self, 'native_access_pair_clearances_nm', ())
        _require(not current or current == frozen,
                 "bound native access pairs cannot be replaced or removed")
        if current == frozen:
            return self
        return replace(self, native_access_pair_clearances_nm=frozen)

    def payload(self):
        return self._payload(self._validated_data())

    def _payload(self, data):
        result = {
            "schema": ("clay.edge-rail-reservation.v9" if getattr(self, 'top_margin_nm', 0)
                       else "clay.edge-rail-reservation.v8" if any(row.wide_space_nm for row in self.demands)
                       else "clay.edge-rail-reservation.v7" if any(row.offset_nm for row in self.demands)
                       else "clay.edge-rail-reservation.v6" if any(row.guard_source_members for row in self.stamps)
                       else "clay.edge-rail-reservation.v5" if data['native_feol_pairs']
                       else "clay.edge-rail-reservation.v4" if data['native_access_pairs']
                       else "clay.edge-rail-reservation.v3" if any(row.access_layer_keys for row in self.demands)
                       else "clay.edge-rail-reservation.v2" if any(row.joint_component_ids for row in self.demands)
                       else "clay.edge-rail-reservation.v1"), "predicate_id": self.predicate_id,
            "source_sha256": self.source_sha256, "topology_sha256": self.topology_sha256,
            "pdk_id": self.pdk_id, "pdk_provenance_sha256": self.pdk_provenance_sha256,
            "manufacturing_grid_nm": self.manufacturing_grid_nm,
            "masters": [list(row) for row in data['masters']],
            "demands": [dict(row) for row in data['demands']],
            "stamps": [dict(row) for row in data['stamps']],
            "atoms": [dict(row) for row in data['atoms']],
        }
        if any(row.access_layer_keys for row in self.demands):
            result['access_predicate_id'] = EDGE_RAIL_ACCESS_PREDICATE_ID
        if any(row.wide_space_nm for row in self.demands):
            result['wide_predicate_id'] = EDGE_RAIL_WIDE_PREDICATE_ID
        if data['native_access_pairs']:
            result['native_access_pair_predicate_id'] = NATIVE_RAIL_ACCESS_PAIR_PREDICATE_ID
            result['native_access_pair_clearances_nm'] = [list(row) for row in data['native_access_pairs']]
        if data['native_feol_pairs']:
            from .native_feol_reservation import NATIVE_FEOL_PREDICATE_ID
            result['native_feol_predicate_id'] = NATIVE_FEOL_PREDICATE_ID
            result['native_feol_pairs'] = [dict(row) for row in data['native_feol_pairs']]
        if getattr(self, 'right_margin_nm', 0):
            result['right_margin_nm'] = self.right_margin_nm
        if getattr(self, 'top_margin_nm', 0):
            result['top_margin_nm'] = self.top_margin_nm
        return result

    def canonical_data(self):
        return self.payload()

    @property
    def sha256(self):
        data = self._validated_data()
        if data['sha256'] is None:
            data['sha256'] = hashlib.sha256(_json(self._payload(data))).hexdigest()
        return data['sha256']

    def lowering_counts(self, *, physical_bbox_nm=None):
        """Derive primitive/guard counts from input data, without reading a model.

        A pair clause counts once in the total and once under each participating
        demand guard. Consumers must not sum guard counts as primitive counts.
        """
        data = self._validated_data()
        total = 8 * (len(self.stamps) + len(self.atoms)) + 4 + 2
        total += 16 if self.manufacturing_grid_nm > 1 else 0
        total += 4 if physical_bbox_nm is not None else 0
        total += 9 * len(data['native_access_pairs'])
        total += 9 * sum(len(cover) for _, _, cover in data['native_feol_covers'])
        guarded = {}
        for demand in self.demands:
            count = 4 + int(demand.terminal_count == 0)
            total += count
            for atom in self.atoms:
                if atom.layer_key == demand.layer_key:
                    total += 19 if atom.net == demand.net else 9
                    count += 1
            for _master, domain in data['domains'][demand.demand_id]:
                total += 9 if _domain_nonempty(domain) else 1
                count += 1
            guarded[demand.demand_id] = count
        for left, right in combinations(self.demands, 2):
            if left.layer_key != right.layer_key:
                continue
            total += (1 if left.actual_axis != right.actual_axis
                      else (19 if left.net == right.net else 9))
            guarded[left.demand_id] += 1
            guarded[right.demand_id] += 1
        return total, tuple(sorted(guarded.items()))

    def _dimensions(self, dims, *, data=None):
        data = self._validated_data() if data is None else data
        expected = data['dimensions']
        _require(isinstance(dims, Mapping) and dict(dims) == expected
                 and all(type(pair) is tuple and all(type(value) is int for value in pair)
                         for pair in dims.values()), "dimension inventory differs from original masters")
        return expected

    def replay(self, *, dims, positions, physical_bbox_nm=None):
        data = self._validated_data()
        expected = self._dimensions(dims, data=data)
        _require(isinstance(positions, Mapping) and set(positions) == set(expected),
                 "position inventory differs from original masters")
        _require(all(type(pair) is tuple and len(pair) == 2
                     and all(type(value) is int and abs(value) <= _LIMIT for value in pair)
                     for pair in positions.values()), "position coordinates must be bounded exact integers")
        for left, right, gap in data['native_access_pairs']:
            lx, ly = positions[left]
            rx, ry = positions[right]
            lw, lh = expected[left]
            rw, rh = expected[right]
            if not (lx + lw + gap <= rx or rx + rw + gap <= lx
                    or ly + lh + gap <= ry or ry + rh + gap <= ly):
                return False
        from .native_feol_reservation import native_feol_rows_satisfied
        if not native_feol_rows_satisfied(getattr(self, 'native_feol_pairs', ()), positions):
            return False
        stamps = tuple(_translated(row.bbox_nm, positions[row.master_id]) for row in self.stamps)
        atoms = tuple(_translated(row.bbox_nm, positions[row.master_id]) for row in self.atoms)
        if any(abs(value) > _LIMIT for box in (*stamps, *atoms) for value in box):
            return False
        right_margin = getattr(self, 'right_margin_nm', 0)
        top_margin = getattr(self, 'top_margin_nm', 0)
        raw = (min(box[0] for box in stamps), min(box[1] for box in stamps),
               max(box[2] for box in stamps) + right_margin, max(box[3] for box in stamps) + top_margin)
        if physical_bbox_nm is not None:
            _box(physical_bbox_nm, "supplied physical bbox")
            if physical_bbox_nm != raw:
                return False
        cell = tuple(_nearest_on_grid(value, self.manufacturing_grid_nm) for value in raw)
        if cell[0] >= cell[2] or cell[1] >= cell[3] or any(abs(value) > _LIMIT for value in cell):
            return False
        stripes = {}
        for demand in self.demands:
            if not demand.terminal_count:
                return False
            stripe = _stripe(cell, demand)
            if not _inside(stripe, cell):
                return False
            for atom, box in zip(self.atoms, atoms):
                if atom.layer_key == demand.layer_key and _conflict(
                        stripe, box, max(demand.min_space_nm, demand.wide_space_nm), atom.net == demand.net):
                    return False
            for master, domain in data['domains'][demand.demand_id]:
                if not _domain_nonempty(domain) or not _domain_exposed(
                        _translated(domain, positions[master]), stripe, demand.access_clearance_nm):
                    return False
            stripes[demand.demand_id] = stripe
        for left, right in combinations(self.demands, 2):
            if left.layer_key != right.layer_key:
                continue
            if left.actual_axis != right.actual_axis:
                return False
            same = left.net == right.net
            gap = max(left.min_space_nm, left.wide_space_nm, right.min_space_nm, right.wide_space_nm,
                      0 if same else self.manufacturing_grid_nm)
            if _conflict(stripes[left.demand_id], stripes[right.demand_id], gap, same):
                return False
        return True

    def add_constraints(self, model, *, xs, ys, dims, physical_bbox_nm=None, assumption_for=None):
        """Append equivalent integer constraints without changing any original rows.

        Optional assumption literals enable per-demand attribution. Outline and
        predicate definitions remain unconditional; each hard demand/pair clause
        is enforced when its corresponding demand assumptions are enabled.
        """
        from ortools.sat.python import cp_model

        data = self._validated_data()
        expected = self._dimensions(dims, data=data)
        _require(isinstance(model, cp_model.CpModel), "CP model has a foreign type")
        _require(isinstance(xs, Mapping) and isinstance(ys, Mapping)
                 and set(xs) == set(ys) == set(expected), "CP coordinate inventory differs")

        def own_variable(variable, label):
            _require(isinstance(variable, cp_model.IntVar), f"{label} must be an own-model integer variable")
            index = variable.Index()
            _require(0 <= index < len(model.Proto().variables)
                     and variable.Proto() is model.Proto().variables[index],
                     f"{label} references a foreign model variable")
            return tuple(model.Proto().variables[index].domain)

        bounds = {}
        for owner in expected:
            for axis, variables in enumerate((xs, ys)):
                variable = variables[owner]
                domain = own_variable(variable, "CP coordinate")
                _require(bool(domain) and abs(domain[0]) <= _LIMIT and abs(domain[-1]) <= _LIMIT,
                         "CP coordinate bounds exceed supported integer geometry")
                bounds[owner, axis] = (domain[0], domain[-1])
        _require(assumption_for is None or callable(assumption_for), "assumption mapping must be callable")
        if physical_bbox_nm is not None:
            _require(type(physical_bbox_nm) is tuple and len(physical_bbox_nm) == 4,
                     "supplied physical bbox must have four expressions")
            for expression in physical_bbox_nm:
                if type(expression) is int:
                    terms, constant = {}, expression
                else:
                    _require(isinstance(expression, cp_model.LinearExpr),
                             "physical bbox requires exact integer linear expressions")
                    try:
                        terms, constant = expression.GetIntegerVarValueMap()
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError("edge rail reservation physical bbox is not exact integer arithmetic") from exc
                _require(type(constant) is int, "physical bbox offset must be an exact integer")
                activity = abs(constant) + _LIMIT
                for variable, coefficient in terms.items():
                    _require(type(coefficient) is int, "physical bbox coefficient must be exact integer")
                    domain = own_variable(variable, "physical bbox")
                    activity += abs(coefficient) * max(abs(domain[0]), abs(domain[-1]))
                _require(activity <= (1 << 62) - 1, "physical bbox activity exceeds int64 safety bound")
        guards = {}
        guard_indices = set()
        for demand in self.demands:
            if assumption_for is None:
                guards[demand.demand_id] = ()
                continue
            guard = assumption_for(demand.demand_id)
            domain = own_variable(guard, "demand assumption")
            index = guard.Index()
            _require(domain == (0, 1) and index in model.Proto().assumptions,
                     "demand assumption must be an active positive Boolean guard")
            _require(index not in guard_indices, "demand assumptions must have distinct guards")
            guard_indices.add(index)
            guards[demand.demand_id] = (guard,)
        prefix = f"edge_rail:{len(model.Proto().variables)}:"
        counter = 0

        def fresh(label):
            nonlocal counter
            counter += 1
            return f"{prefix}{label}:{counter}"

        def boolean_le(left, right):
            flag = model.NewBoolVar(fresh("le"))
            model.Add(left <= right).OnlyEnforceIf(flag)
            model.Add(left >= right + 1).OnlyEnforceIf(flag.Not())
            return flag

        def clause(literals, guards):
            result = model.AddBoolOr(literals)
            if guards:
                result.OnlyEnforceIf(guards)

        def separated(left, right, gap, same_net, guards):
            # A conductor may lie beyond the stamp outline, so all four ways of
            # obtaining enough spacing matter, including the rail's run axis.
            threshold = max(1, gap)
            far = [boolean_le(left[2] + threshold, right[0]),
                   boolean_le(right[2] + threshold, left[0]),
                   boolean_le(left[3] + threshold, right[1]),
                   boolean_le(right[3] + threshold, left[1])]
            if same_net:
                touches = [boolean_le(left[0], right[2]), boolean_le(right[0], left[2]),
                           boolean_le(left[1], right[3]), boolean_le(right[1], left[3])]
                touch = model.NewBoolVar(fresh("touch"))
                model.AddBoolAnd(touches).OnlyEnforceIf(touch)
                model.AddBoolOr([item.Not() for item in touches]).OnlyEnforceIf(touch.Not())
                far.append(touch)
            clause(far, guards)

        def bounded_box(row):
            box = _translated(row.bbox_nm, (xs[row.master_id], ys[row.master_id]))
            for value in box:
                model.Add(value >= -_LIMIT)
                model.Add(value <= _LIMIT)
            return box

        stamps = tuple(bounded_box(row) for row in self.stamps)
        atoms = tuple(bounded_box(row) for row in self.atoms)
        right_margin = getattr(self, 'right_margin_nm', 0)
        top_margin = getattr(self, 'top_margin_nm', 0)
        raw = []
        for coordinate in range(4):
            axis = coordinate % 2
            margin = right_margin if coordinate == 2 else (top_margin if coordinate == 3 else 0)
            low = min(bounds[row.master_id, axis][0] + row.bbox_nm[coordinate] + margin for row in self.stamps)
            high = max(bounds[row.master_id, axis][1] + row.bbox_nm[coordinate] + margin for row in self.stamps)
            # Keep a nonempty arithmetic domain even when every possible stamp
            # is outside the supported world. The explicit stamp bounds above
            # then prove infeasibility without constructing an invalid model.
            variable = model.NewIntVar(low, high, fresh("outline"))
            if coordinate < 2:
                model.AddMinEquality(variable, [box[coordinate] for box in stamps])
            else:
                model.AddMaxEquality(variable, [box[coordinate] + margin for box in stamps])
            raw.append(variable)
        if physical_bbox_nm is not None:
            for actual, supplied in zip(raw, physical_bbox_nm):
                model.Add(actual == supplied)
        grid = self.manufacturing_grid_nm
        cell = []
        for value in raw:
            if grid == 1:
                cell.append(value)
                continue
            # floor((coordinate + floor(grid/2))/grid): ties go toward +inf,
            # matching emitted outline snapping for both coordinate signs.
            quotient = model.NewIntVar((-_LIMIT + grid // 2) // grid,
                (_LIMIT + grid // 2) // grid, fresh("grid"))
            model.Add(value + grid // 2 >= grid * quotient)
            model.Add(value + grid // 2 <= grid * quotient + grid - 1)
            model.Add(grid * quotient >= -_LIMIT)
            model.Add(grid * quotient <= _LIMIT)
            cell.append(grid * quotient)
        model.Add(cell[0] < cell[2])
        model.Add(cell[1] < cell[3])
        stripes = {}
        for demand in self.demands:
            enabled = guards[demand.demand_id]
            if not demand.terminal_count:
                clause([], enabled)
            stripe = _stripe(cell, demand)
            stripes[demand.demand_id] = stripe
            for left, right in ((cell[0], stripe[0]), (cell[1], stripe[1]),
                                (stripe[2], cell[2]), (stripe[3], cell[3])):
                constraint = model.Add(left <= right)
                if enabled:
                    constraint.OnlyEnforceIf(enabled)
            for atom, box in zip(self.atoms, atoms):
                if atom.layer_key == demand.layer_key:
                    separated(stripe, box, max(demand.min_space_nm, demand.wide_space_nm), atom.net == demand.net, enabled)
            for master, domain in data['domains'][demand.demand_id]:
                if not _domain_nonempty(domain):
                    clause([], enabled)
                    continue
                gap = max(1, demand.access_clearance_nm)
                box = _translated(domain, (xs[master], ys[master]))
                clause((boolean_le(stripe[2] + gap, box[2]),
                        boolean_le(box[0] + gap, stripe[0]),
                        boolean_le(stripe[3] + gap, box[3]),
                        boolean_le(box[1] + gap, stripe[1])), enabled)
        for left, right in combinations(self.demands, 2):
            if left.layer_key != right.layer_key:
                continue
            enabled = (*guards[left.demand_id], *guards[right.demand_id])
            if left.actual_axis != right.actual_axis:
                clause([], enabled)
                continue
            same = left.net == right.net
            separated(stripes[left.demand_id], stripes[right.demand_id],
                max(left.min_space_nm, left.wide_space_nm, right.min_space_nm, right.wide_space_nm,
                    0 if same else grid), same, enabled)
        # Independent of current-layer collision exclusions and rail guards:
        # these source-derived pair obligations reserve future native access.
        for left, right, gap in data['native_access_pairs']:
            lw, lh = dims[left]
            rw, rh = dims[right]
            separated((xs[left], ys[left], xs[left] + lw, ys[left] + lh),
                      (xs[right], ys[right], xs[right] + rw, ys[right] + rh),
                      gap, False, ())
        for left, right, cover in data['native_feol_covers']:
            dx, dy = xs[right] - xs[left], ys[right] - ys[left]
            for x0, y0, x1, y1 in cover:
                clause((boolean_le(dx, x0 - 1), boolean_le(x1, dx),
                        boolean_le(dy, y0 - 1), boolean_le(y1, dy)), ())


def _lower_access_domains(atoms, demand):
    """Necessary escape domains inside actual foreign closed metal rings.

    A covered pad is not itself a routing obstruction: an M1 stub can extend
    before rising. Only a closed foreign M1 ring bounds that motion. Its interior
    must retain at least one possible via centre beyond an upper foreign stripe.
    The interior bounding box is an optimistic domain, not a routability claim.
    All actual obstacles, via landings and full connectivity remain router gates.
    """
    if not demand.access_layer_keys:
        return ()
    import klayout.db as kdb
    from clay.layout.model import ROUTING_ABSTRACTION_KINDS

    rings, terminals = {}, {}
    for atom in atoms:
        if atom.layer_key not in demand.access_layer_keys or atom.kind in ROUTING_ABSTRACTION_KINDS:
            continue
        if atom.kind == 'guard-ring-m1':
            key = (atom.master_id, atom.stamp_id, atom.layer_key, atom.net)
            rings.setdefault(key, []).append(atom.bbox_nm)
        if atom.net is not None and atom.net != demand.net:
            terminals.setdefault((atom.master_id, atom.layer_key), []).append(atom)
    result = set()
    margin = demand.access_clearance_nm
    for (master, _stamp, layer, ring_net), boxes in sorted(rings.items(), key=repr):
        region = kdb.Region()
        for box in boxes:
            region.insert(kdb.Box(*box))
        for polygon in region.merged().each():
            for hole_index in range(polygon.holes()):
                hole = kdb.Polygon(list(polygon.each_point_hole(hole_index)))
                extent = hole.bbox()
                box = (extent.left, extent.bottom, extent.right, extent.top)
                trapped = any(atom.net != ring_net and
                    box[0] <= atom.bbox_nm[0] and box[1] <= atom.bbox_nm[1] and
                    atom.bbox_nm[2] <= box[2] and atom.bbox_nm[3] <= box[3]
                    for atom in terminals.get((master, layer), ()))
                if trapped:
                    result.add((master, (box[0] + margin, box[1] + margin,
                                         box[2] - margin, box[3] - margin)))
    return tuple(sorted(result))


def _domain_nonempty(box):
    return box[0] <= box[2] and box[1] <= box[3]


def _domain_exposed(box, stripe, gap):
    gap = max(1, gap)
    return (stripe[2] + gap <= box[2] or box[0] + gap <= stripe[0]
            or stripe[3] + gap <= box[3] or box[1] + gap <= stripe[1])


def _translated(box, origin):
    return (origin[0] + box[0], origin[1] + box[1], origin[0] + box[2], origin[1] + box[3])


def _stripe(cell, demand):
    x0, y0, x1, y1 = cell
    width = demand.width_nm
    offset = demand.offset_nm
    return {"top": (x0, y1 - width - offset, x1, y1 - offset),
            "bottom": (x0, y0 + offset, x1, y0 + offset + width),
            "left": (x0 + offset, y0, x0 + offset + width, y1),
            "right": (x1 - width - offset, y0, x1 - offset, y1)}[demand.edge]


def _inside(box, cell):
    return (cell[0] <= box[0] < box[2] <= cell[2]
            and cell[1] <= box[1] < box[3] <= cell[3])


def _conflict(left, right, spacing, same_net):
    return (_rects_within_spacing(left, right, spacing)
            and not (same_net and _rects_touch_or_overlap(left, right)))


def add_edge_rail_constraints(model, *, xs, ys, dims, reservation,
                              physical_bbox_nm=None, assumption_for=None):
    _require(type(reservation) is EdgeRailReservationIR, "IR has a foreign type")
    reservation.add_constraints(model, xs=xs, ys=ys, dims=dims,
        physical_bbox_nm=physical_bbox_nm, assumption_for=assumption_for)


def replay_edge_rail_reservation(*, positions, dims, reservation, physical_bbox_nm=None):
    _require(type(reservation) is EdgeRailReservationIR, "IR has a foreign type")
    return reservation.replay(dims=dims, positions=positions, physical_bbox_nm=physical_bbox_nm)


__all__ = ["EDGE_RAIL_PREDICATE_ID", "EdgeRailConductorAtom", "EdgeRailDemandIR",
           "EdgeRailReservationIR", "EdgeRailStampIR", "add_edge_rail_constraints",
           "replay_edge_rail_reservation"]
