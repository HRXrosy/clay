"""Observed, source-bound route resources in an imported layout.

This deliberately does not construct a normal ``route_cell`` result or a
signoff verdict.  It owns a restricted subtractive operation on independently
observed rectangular metal tails.  Full device equivalence and foundry rules
remain obligations of the native LVS/DRC flow.  The caller supplies a frozen
PDK layer/connectivity table; there are no process or circuit-name defaults.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import klayout.db as kdb

from clay.layout.spatial.extract import extract_canonical_layout_from_gds


class ImportedRouteUnsupported(ValueError):
    """The requested geometry/ownership is outside the proved import scope."""


def _require(condition, message):
    if not condition:
        raise ImportedRouteUnsupported(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_value(value, name):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name} requires a lowercase SHA-256")
    return value


def _layer(key):
    _require(type(key) in (tuple, list) and len(key) == 2
             and all(type(v) is int and 0 <= v <= 65535 for v in key),
             "layer identity requires two exact nonnegative integers")
    return tuple(key)


def _box_tuple(box):
    return box.left, box.bottom, box.right, box.top


def _bbox_overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _poly(shape):
    if shape.is_box():
        return kdb.Polygon(shape.box)
    if shape.is_polygon():
        return shape.polygon
    if shape.is_path():
        return shape.path.polygon()
    return None


def _shape_data(shape):
    # Keep paths and texts as original objects, not only their polygon masks.
    return (shape.to_s(), sorted((str(k), repr(v)) for k, v in shape.properties().items()))


def _inventory(layout):
    result = {}
    for cell in layout.each_cell():
        shapes = []
        for index in layout.layer_indices():
            info = layout.get_info(index)
            shapes.extend(((info.layer, info.datatype), _shape_data(s))
                          for s in cell.shapes(index).each())
        instances = [(inst.cell.name, inst.cell_inst.cplx_trans.to_s(),
                      (inst.cell_inst.a.x, inst.cell_inst.a.y),
                      (inst.cell_inst.b.x, inst.cell_inst.b.y),
                      inst.cell_inst.na, inst.cell_inst.nb,
                      sorted((str(k), repr(v)) for k, v in inst.properties().items()))
                     for inst in cell.each_inst()]
        result[cell.name] = dict(shapes=sorted(shapes, key=_canonical),
                                 instances=sorted(instances, key=_canonical),
                                 properties=sorted((str(k), repr(v)) for k, v in cell.properties().items()))
    return result


@dataclass(frozen=True)
class ImportedRouteResource:
    resource_id: str
    net: str
    layer: str
    layer_key: tuple[int, int]
    cell_name: str
    occurrence_path: str
    shape_sha256: str
    component_sha256: str
    local_box_nm: tuple[int, int, int, int]
    global_box_nm: tuple[int, int, int, int]
    end: str
    max_trim_nm: int
    local_labels: tuple[str, ...]
    global_labels: tuple[str, ...]

    def canonical_data(self):
        return asdict(self)


class ImportedRouteObservation:
    __slots__ = ("_path", "_config", "_resources", "_data", "_digest")

    def __init__(self, *args, **kwargs):
        raise TypeError("use observe_imported_route_resources to produce geometry authority")

    @property
    def resources(self):
        return self._resources

    @property
    def sha256(self):
        return self._digest

    def canonical_data(self):
        return json.loads(_canonical(self._data))

    def require_current(self):
        _require(_sha(self._path) == self._config["expected_gds_sha256"],
                 "imported source GDS changed after observation")
        _require(_digest(self._data) == self._digest, "imported observation data changed")
        _require(_sha(__file__) == self._data["producer_sha256"], "imported geometry producer changed after observation")
        _require([r.canonical_data() for r in self._resources] == self._data["resources"],
                 "imported resource inventory changed")


@dataclass(frozen=True)
class ImportedRouteArtifact:
    gds_path: str
    gds_sha256: str
    source_sha256: str
    source_binding_sha256: str
    observation_sha256: str
    proof: dict

    def canonical_data(self):
        return asdict(self)


def _normalize_config(*, top_cell, expected_gds_sha256, pdk_id, metal_layers,
                      connector_layers, terminal_layers, label_layers,
                      net_label_mapping, expected_nets, source_binding_sha256,
                      manufacturing_grid_nm, editable_layers):
    _require(type(top_cell) is str and top_cell, "explicit top cell is required")
    _require(type(pdk_id) is str and pdk_id, "explicit PDK identity is required")
    _require(type(manufacturing_grid_nm) is int and manufacturing_grid_nm > 0,
             "manufacturing grid requires an exact positive integer nm")
    _require(all(type(name) is str and name for name in metal_layers),
             "conductive layer names require exact nonempty strings")
    metals = tuple(sorted((name, _layer(key)) for name, key in metal_layers.items()))
    _require(metals and len({key for _, key in metals}) == len(metals),
             "conductive layer identities must be nonempty and unique")
    names = {name for name, _ in metals}
    editable = tuple(sorted(names if editable_layers is None else editable_layers))
    _require(editable and len(editable) == len(set(editable)) and set(editable) <= names,
             "editable layers must be a nonempty subset of supplied conductive layers")
    connectors = tuple(sorted((_layer(key), tuple(pair)) for key, pair in connector_layers.items()))
    _require(all(len(pair) == 2 and pair[0] != pair[1] and set(pair) <= names
                 for _, pair in connectors), "connector references absent or identical conductive layers")
    _require(not ({key for key, _ in connectors} & {key for _, key in metals}),
             "connector and conductor layers must be distinct")
    labels = tuple(sorted((_layer(key), name) for key, name in label_layers.items()))
    _require(labels and all(name in names for _, name in labels),
             "label layers must bind explicit conductive layers")
    mapping = tuple(sorted(net_label_mapping.items()))
    _require(mapping and all(type(k) is str and k and type(v) is str and v for k, v in mapping),
             "explicit native-label to source-net mapping is required")
    targets = tuple(sorted(expected_nets))
    _require(targets and len(set(targets)) == len(targets)
             and all(type(n) is str and n for n in targets), "target nets must be nonempty and unique")
    _require(set(v for _, v in mapping) == set(targets),
             "label mapping and explicit target net denominator differ")
    return dict(top_cell=top_cell,
                expected_gds_sha256=_sha_value(expected_gds_sha256, "GDS binding"),
                pdk_id=pdk_id, metal_layers=metals, connector_layers=connectors,
                terminal_layers=tuple(sorted({_layer(k) for k in terminal_layers})),
                label_layers=labels, net_label_mapping=mapping, expected_nets=targets,
                source_binding_sha256=_sha_value(source_binding_sha256, "source binding"),
                manufacturing_grid_nm=manufacturing_grid_nm, editable_layers=editable)


def _load(path, config):
    _require(_sha(path) == config["expected_gds_sha256"], "imported GDS SHA-256 differs from source")
    layout = kdb.Layout()
    layout.read(str(path))
    _require(not layout.properties(), "library-level GDS properties are outside the preserved import scope")
    top = layout.cell(config["top_cell"])
    _require(top is not None, "explicit imported top cell is absent")
    unit = Decimal(str(layout.dbu)) * 1000
    _require(unit > 0 and unit == unit.to_integral_value(),
             "only integer-nanometre GDS database units are supported")
    return layout, top, int(unit)


def _occurrences(layout, top):
    occurrences = []

    def visit(cell, transform, path, ancestry):
        _require(cell.cell_index() not in ancestry, "cyclic imported hierarchy is unsupported")
        occurrences.append((cell, transform, path))
        children = sorted(cell.each_inst(), key=lambda i: (i.cell.name, i.cell_inst.to_s()))
        for index, inst in enumerate(children):
            for member, child_transform in enumerate(inst.cell_inst.each_cplx_trans()):
                _require(not child_transform.is_complex(),
                         "scaled or non-orthogonal imported hierarchy is unsupported")
                child_path = f"{path}/{inst.cell.name}[{index},{member}]"
                visit(inst.cell, transform * child_transform, child_path,
                      ancestry | {cell.cell_index()})
    visit(top, kdb.ICplxTrans(), top.name, set())
    return occurrences


def _read_geometry(layout, top, config):
    occurrences = _occurrences(layout, top)
    rows = defaultdict(list)
    labels = []
    label_layers = dict(config["label_layers"])
    for cell, trans, path in occurrences:
        for index in layout.layer_indices():
            info = layout.get_info(index)
            key = (info.layer, info.datatype)
            for shape in cell.shapes(index).each():
                polygon = _poly(shape)
                if polygon is not None:
                    global_poly = polygon.transformed(trans)
                    rows[key].append(dict(cell=cell.name, path=path, trans=trans,
                                          shape=shape, local=polygon, poly=global_poly,
                                          bbox=_box_tuple(global_poly.bbox()),
                                          shape_data=_shape_data(shape)))
                elif shape.is_text() and key in label_layers:
                    text = shape.text
                    point = trans * kdb.Point(text.trans.disp.x, text.trans.disp.y)
                    labels.append(dict(local=text.string, global_name=path + "/" + text.string,
                                       path=path, layer=label_layers[key],
                                       x=point.x, y=point.y))
    return occurrences, rows, labels


def _components(rows, labels, config):
    """Exact merged conductor components joined only by supplied PDK cuts."""
    nodes = []
    by_layer = defaultdict(list)
    for name, key in config["metal_layers"]:
        region = kdb.Region()
        for row in rows[key]:
            region.insert(row["poly"])
        for poly in region.merged().each():
            index = len(nodes)
            nodes.append(dict(layer=name, key=key, poly=poly, region=kdb.Region(poly),
                              bbox=_box_tuple(poly.bbox())))
            by_layer[name].append(index)
    parent = list(range(len(nodes)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def hits(name, poly, bbox):
        region = kdb.Region(poly)
        return [i for i in by_layer[name] if _bbox_overlap(nodes[i]["bbox"], bbox)
                and not (nodes[i]["region"] & region).is_empty()]

    for key, (lower, upper) in config["connector_layers"]:
        for cut in rows[key]:
            low = hits(lower, cut["poly"], cut["bbox"])
            high = hits(upper, cut["poly"], cut["bbox"])
            # A floating cut does not invent connectivity. All cut geometry is
            # preserved and full DRC, not this observer, decides its legality.
            if low and high:
                joined = low + high
                first = root(joined[0])
                for other in joined[1:]:
                    parent[root(other)] = first
    groups = defaultdict(list)
    for index in range(len(nodes)):
        groups[root(index)].append(index)
    components = []
    node_component = {}
    for members in groups.values():
        descriptor = sorted((nodes[i]["layer"], nodes[i]["poly"].to_s()) for i in members)
        component = dict(nodes=tuple(members), sha256=_digest(descriptor), labels=[])
        for i in members:
            node_component[i] = len(components)
        components.append(component)
    mapping = dict(config["net_label_mapping"])
    net_components = defaultdict(set)
    for label in labels:
        point = kdb.Point(label["x"], label["y"])
        found = {node_component[i] for i in by_layer[label["layer"]]
                 if nodes[i]["poly"].inside(point)}
        label["components"] = tuple(sorted(found))
        target = mapping.get(label["global_name"], mapping.get(label["local"]))
        if target is not None:
            _require(len(found) == 1, f"mapped label {label['global_name']} has ambiguous or absent conductor ownership")
            net_components[target].update(found)
        for index in found:
            components[index]["labels"].append(label)
    for net in config["expected_nets"]:
        _require(len(net_components[net]) == 1, f"source net {net} lacks one unique native label component")
    _require(len({next(iter(value)) for value in net_components.values()}) == len(net_components),
             "different source nets share a conductive component")
    component_nets = {next(iter(value)): net for net, value in net_components.items()}
    return nodes, components, by_layer, node_component, component_nets


def _clip_attachment(box, polygon):
    bbox = _box_tuple(polygon.bbox())
    if not _bbox_overlap(box, bbox):
        return None
    region = kdb.Region(kdb.Box(*box))
    other = kdb.Region(polygon)
    overlap = region & other
    if not overlap.is_empty():
        return _box_tuple(overlap.bbox())
    if region.interacting(other).is_empty():
        return None
    return max(box[0], bbox[0]), max(box[1], bbox[1]), min(box[2], bbox[2]), min(box[3], bbox[3])


def _observe(path, config):
    layout, top, unit = _load(path, config)
    occurrences, rows, labels = _read_geometry(layout, top, config)
    nodes, components, by_layer, node_component, component_nets = _components(rows, labels, config)
    # A shared master under another root is still shared: a local edit must
    # never silently change an unselected occurrence outside the selected top.
    occurrence_count = Counter(cell.name for root in layout.top_cells()
                               for cell, _, _ in _occurrences(layout, root))
    protected_keys = set(config["terminal_layers"]) | {key for key, _ in config["connector_layers"]}
    protected = [row for key in protected_keys for row in rows[key]]
    resources, bindings, rejected = [], {}, Counter()
    for layer_name, key in config["metal_layers"]:
        if layer_name not in config["editable_layers"]:
            continue
        for row in rows[key]:
            if not row["local"].is_box() or row["shape"].is_path():
                rejected["nonrectangular_or_path"] += 1
                continue
            box = row["bbox"]
            width, height = box[2] - box[0], box[3] - box[1]
            if width == height:
                continue
            owners = {node_component[i] for i in by_layer[layer_name]
                      if _bbox_overlap(nodes[i]["bbox"], box)
                      and not (nodes[i]["region"] & kdb.Region(row["poly"])).is_empty()}
            if len(owners) != 1 or next(iter(owners)) not in component_nets:
                continue
            component_id = next(iter(owners))
            if occurrence_count[row["cell"]] != 1:
                rejected["shared_master_requires_occurrence_specialization"] += 1
                continue
            identical = [other for other in rows[key] if other["cell"] == row["cell"]
                         and other["shape_data"] == row["shape_data"]]
            if len(identical) != 1:
                rejected["duplicate_local_shape"] += 1
                continue
            attachments = []
            for other in rows[key]:
                if other is row:
                    continue
                attached = _clip_attachment(box, other["poly"])
                if attached is not None:
                    attachments.append(attached)
            for other in protected:
                attached = _clip_attachment(box, other["poly"])
                if attached is not None:
                    attachments.append(attached)
            # Labels are protected even if not among the selected source nets.
            for label in labels:
                if label["layer"] == layer_name and box[0] <= label["x"] <= box[2] and box[1] <= label["y"] <= box[3]:
                    attachments.append((label["x"], label["y"], label["x"], label["y"]))
            if not attachments:
                rejected["unanchored_rectangle"] += 1
                continue
            if width > height:
                ends = (("left", min(a[0] for a in attachments) - box[0]),
                        ("right", box[2] - max(a[2] for a in attachments)))
                residual = width - height
            else:
                ends = (("bottom", min(a[1] for a in attachments) - box[1]),
                        ("top", box[3] - max(a[3] for a in attachments)))
                residual = height - width
            for end, extent in ends:
                maximum = min(extent, residual) * unit
                grid = config["manufacturing_grid_nm"]
                maximum -= maximum % grid
                if maximum <= 0:
                    continue
                net = component_nets[component_id]
                component = components[component_id]
                shape_sha = _digest(row["shape_data"])
                identity = dict(gds=config["expected_gds_sha256"], pdk=config["pdk_id"],
                                config=_digest(config), cell=row["cell"], path=row["path"],
                                shape=shape_sha, layer=key, end=end, net=net,
                                component=component["sha256"])
                resource = ImportedRouteResource(
                    "tail_" + _digest(identity), net, layer_name, key, row["cell"], row["path"],
                    shape_sha, component["sha256"],
                    tuple(v * unit for v in _box_tuple(row["local"].bbox())),
                    tuple(v * unit for v in box), end, maximum,
                    tuple(sorted({label["local"] for label in component["labels"]})),
                    tuple(sorted({label["global_name"] for label in component["labels"]})))
                resources.append(resource)
                bindings[resource.resource_id] = row
    resources = tuple(sorted(resources, key=lambda r: r.resource_id))
    authority = extract_canonical_layout_from_gds(path, top_cell=top.name,
                                                 expected_source_sha256=config["expected_gds_sha256"])
    data = dict(schema="clay.imported-route-observation/v1", scope="imported_rectangular_conductor_tails",
                producer_sha256=_sha(__file__),
                source_gds_sha256=config["expected_gds_sha256"], source_binding_sha256=config["source_binding_sha256"],
                layer_config_sha256=_digest(config), top_cell=top.name, pdk_id=config["pdk_id"],
                database_unit_nm=unit, manufacturing_grid_nm=config["manufacturing_grid_nm"],
                canonical_mask_sha256=authority.mask_view.sha256,
                canonical_construction_sha256=authority.construction_view.sha256,
                local_object_inventory_sha256=_digest(_inventory(layout)),
                conductive_component_count=len(components),
                target_components={net: components[index]["sha256"] for index, net in component_nets.items()},
                resources=[r.canonical_data() for r in resources], skipped_scope_counts=dict(rejected),
                signoff="NOT_ESTABLISHED")
    observation = object.__new__(ImportedRouteObservation)
    observation._path, observation._config = Path(path), config
    observation._resources, observation._data = resources, data
    observation._digest = _digest(data)
    observation.require_current()
    return observation, (layout, top, unit, rows, labels, nodes, components, bindings)


def observe_imported_route_resources(gds_path, *, top_cell, expected_gds_sha256,
                                    pdk_id, metal_layers: Mapping[str, tuple[int, int]],
                                    connector_layers: Mapping[tuple[int, int], tuple[str, str]],
                                    terminal_layers: Sequence[tuple[int, int]],
                                    label_layers: Mapping[tuple[int, int], str],
                                    net_label_mapping: Mapping[str, str], expected_nets: Sequence[str],
                                    source_binding_sha256, manufacturing_grid_nm,
                                    editable_layers: Sequence[str] | None = None):
    """Discover eligible tails from a complete, immutable imported GDS.

    Label selectors may be an exact ``occurrence_path/local_label`` or a bare
    local label. Bare labels must all resolve to the same component for the
    selected source net. Unselected internal labels need not be unique.
    ``terminal_layers`` protects supplied PDK terminal-access layers by their
    actual projected geometry; connector cuts are always protected too.
    ``editable_layers`` can restrict editing to selected conductors while the
    complete conductive-layer population remains in the partition proof.
    """
    config = _normalize_config(top_cell=top_cell, expected_gds_sha256=expected_gds_sha256,
                               pdk_id=pdk_id, metal_layers=metal_layers,
                               connector_layers=connector_layers, terminal_layers=terminal_layers,
                               label_layers=label_layers, net_label_mapping=net_label_mapping,
                               expected_nets=expected_nets, source_binding_sha256=source_binding_sha256,
                               manufacturing_grid_nm=manufacturing_grid_nm, editable_layers=editable_layers)
    return _observe(Path(gds_path).resolve(strict=True), config)[0]


def _regions(rows):
    result = {}
    for key, values in rows.items():
        region = kdb.Region()
        for row in values:
            region.insert(row["poly"])
        result[key] = region.merged()
    return result


def _prove_partition(before_nodes, before_components, after_nodes, after_components):
    old_owner = {i: c for c, component in enumerate(before_components) for i in component["nodes"]}
    new_owner = {i: c for c, component in enumerate(after_components) for i in component["nodes"]}
    relation = defaultdict(set)
    for j, node in enumerate(after_nodes):
        for i, original in enumerate(before_nodes):
            if node["layer"] == original["layer"] and _bbox_overlap(node["bbox"], original["bbox"]):
                if not (node["region"] & original["region"]).is_empty():
                    relation[old_owner[i]].add(new_owner[j])
    _require(len(before_components) == len(after_components)
             and set(relation) == set(range(len(before_components)))
             and all(len(v) == 1 for v in relation.values())
             and len({next(iter(v)) for v in relation.values()}) == len(after_components),
             "tail edit changed the complete conductive-component partition")
    return {old: next(iter(new)) for old, new in relation.items()}


def materialize_imported_route_resources(observation, *, trims: Mapping[str, int],
                                        source_sha256, output_gds):
    """Reconstruct declared tails, prove the exact delta, then save a new GDS.

    Zero trims are allowed for a reproducible baseline. Positive trims must
    identify resources actually minted by this observation. Output replacement
    is forbidden. Every nonedited local shape, text, instance and property is
    compared, including objects outside the selected top's reachable hierarchy.
    """
    _require(type(observation) is ImportedRouteObservation, "owned imported observation is required")
    observation.require_current()
    source_sha256 = _sha_value(source_sha256, "candidate source")
    output = Path(output_gds).resolve()
    _require(output != observation._path and not output.exists(), "output GDS must be a new path")
    _require(isinstance(trims, Mapping) and all(type(k) is str and type(v) is int and v >= 0
                                               for k, v in trims.items()),
             "trims require resource IDs and exact nonnegative integer nanometres")
    fresh, state = _observe(observation._path, observation._config)
    _require(fresh.sha256 == observation.sha256, "current imported geometry/resource observation differs")
    layout, top, unit, rows, labels, nodes, components, bindings = state
    resources = {r.resource_id: r for r in fresh.resources}
    _require(set(trims) <= set(resources), "source selects an absent or unsupported route resource")
    before_inventory = _inventory(layout)
    expected_inventory = json.loads(_canonical(before_inventory))
    expected_removed = defaultdict(kdb.Region)
    changed_shapes = set()
    applied = []
    for resource_id, trim_nm in sorted(trims.items()):
        resource = resources[resource_id]
        _require(trim_nm <= resource.max_trim_nm and trim_nm % unit == 0
                 and trim_nm % observation._config["manufacturing_grid_nm"] == 0,
                 "trim exceeds proved tail extent or violates the exact grid")
        if trim_nm == 0:
            continue
        row = bindings[resource_id]
        shape_key = (row["cell"], resource.layer_key, resource.shape_sha256)
        _require(shape_key not in changed_shapes, "simultaneous edits of one original shape are unsupported")
        changed_shapes.add(shape_key)
        box = list(row["bbox"])
        side = {"left": 0, "bottom": 1, "right": 2, "top": 3}[resource.end]
        box[side] += (trim_nm // unit) * (1 if side < 2 else -1)
        _require(box[side] * unit % observation._config["manufacturing_grid_nm"] == 0,
                 "new line end is off the manufacturing grid")
        replacement_global = kdb.Polygon(kdb.Box(*box))
        removed = kdb.Region(row["poly"]) - kdb.Region(replacement_global)
        _require(not removed.is_empty(), "positive trim has no exclusive removed geometry")
        # Replay every protection on the requested exact delta; discovery is
        # not accepted as a substitute for construction-time checking.
        for other in rows[resource.layer_key]:
            if other is not row:
                _require((removed & kdb.Region(other["poly"])).is_empty(),
                         "trim overlaps another same-layer shape")
        protect = set(observation._config["terminal_layers"]) | {k for k, _ in observation._config["connector_layers"]}
        for key in protect:
            for other in rows[key]:
                _require((removed & kdb.Region(other["poly"])).is_empty(),
                         "trim removes connector or terminal access coverage")
        for label in labels:
            if label["layer"] == resource.layer and row["poly"].inside(kdb.Point(label["x"], label["y"])):
                _require(replacement_global.inside(kdb.Point(label["x"], label["y"])),
                         "trim removes a label's conductor coverage")
        local = replacement_global.transformed(row["trans"].inverted())
        _require(local.is_box(), "inverse imported transform is not a Manhattan rectangle")
        cell = layout.cell(row["cell"])
        shape = row["shape"]
        props = shape.properties()
        old_item = json.loads(_canonical((resource.layer_key, _shape_data(shape))))
        expected_inventory[cell.name]["shapes"].remove(old_item)
        cell.shapes(layout.layer(*resource.layer_key)).erase(shape)
        new_shape = cell.shapes(layout.layer(*resource.layer_key)).insert(local.bbox())
        for key, value in props.items():
            new_shape.set_property(key, value)
        expected_inventory[cell.name]["shapes"].append(
            json.loads(_canonical((resource.layer_key, _shape_data(new_shape)))))
        expected_inventory[cell.name]["shapes"].sort(key=_canonical)
        expected_removed[resource.layer_key] += removed
        applied.append(dict(resource_id=resource_id, net=resource.net, trim_nm=trim_nm,
                            removed_area_nm2=int(removed.area()) * unit * unit))
    # Observations are invalidated by changes to the original file even when
    # the original happened to be restored to a geometrically equivalent file.
    observation.require_current()
    _require(_canonical(_inventory(layout)) == _canonical(expected_inventory),
             "reconstruction changed an unselected local object")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clay-imported-route-", dir=output.parent) as directory:
        temporary = Path(directory) / "verified.gds"
        layout.write(str(temporary))
        emitted = kdb.Layout()
        emitted.read(str(temporary))
        _require(_canonical(_inventory(emitted)) == _canonical(expected_inventory),
                 "GDS serialization changed local geometry, labels, instances or properties")
        _, after_rows, after_labels = _read_geometry(emitted, emitted.cell(top.name), observation._config)
        before_regions, after_regions = _regions(rows), _regions(after_rows)
        layer_proofs = []
        for key in sorted(set(before_regions) | set(after_regions)):
            before = before_regions.get(key, kdb.Region())
            after = after_regions.get(key, kdb.Region())
            expected = expected_removed.get(key, kdb.Region()).merged()
            _require((after - before).is_empty() and ((before - after) ^ expected).is_empty(),
                     f"exact full-mask difference differs on layer {key}")
            layer_proofs.append(dict(layer=list(key), removed_area_nm2=int(expected.area()) * unit * unit,
                                     added_area_nm2=0))
        after_nodes, after_components, _, _, _ = _components(after_rows, after_labels, observation._config)
        partition = _prove_partition(nodes, components, after_nodes, after_components)
        before_label_partition = sorted((r["global_name"], r["layer"], r["x"], r["y"],
                                         tuple(sorted(partition[c] for c in r["components"]))) for r in labels)
        after_label_partition = sorted((r["global_name"], r["layer"], r["x"], r["y"], r["components"])
                                       for r in after_labels)
        _require(before_label_partition == after_label_partition, "label ownership partition changed")
        authority = extract_canonical_layout_from_gds(temporary, top_cell=top.name)
        observation.require_current()
        # Exclusive creation: a racing caller cannot turn this into overwrite.
        data = temporary.read_bytes()
        with output.open("xb") as stream:
            stream.write(data)
    proof = dict(schema="clay.imported-route-materialization/v1", scope="imported_rectangular_conductor_tails",
                 source_gds_sha256=observation._config["expected_gds_sha256"],
                 pdk_id=observation._config["pdk_id"], layer_config_sha256=_digest(observation._config),
                 applied=applied, all_layer_differences=layer_proofs,
                 conductive_components_before=len(components), conductive_components_after=len(after_components),
                 component_partition_preserved=True, label_ownership_preserved=True,
                 nonedited_local_objects_preserved=True, connector_and_terminal_coverage_preserved=True,
                 canonical_mask_sha256=authority.mask_view.sha256,
                 canonical_construction_sha256=authority.construction_view.sha256,
                 source_binding_scope="caller_bound_native_topology_and_reference_authority",
                 signoff="NOT_ESTABLISHED", normal_route_cell_result=False)
    return ImportedRouteArtifact(str(output), hashlib.sha256(data).hexdigest(), source_sha256,
                                 observation._config["source_binding_sha256"], observation.sha256, proof)


__all__ = ["ImportedRouteUnsupported", "ImportedRouteResource", "ImportedRouteObservation",
           "ImportedRouteArtifact", "observe_imported_route_resources", "materialize_imported_route_resources"]
