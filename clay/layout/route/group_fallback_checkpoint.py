"""Snapshot and restore transactional group-route state without rewinding budgets."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math

from clay.layout.route.engine import Pin
from clay.layout.route.engine_gridless import _SpatialIndex, _ViaSite, _WireSeg
from clay.layout.route.group_ports import GroupRouteTerminalAccess

_PINNED = (Pin, GroupRouteTerminalAccess, _WireSeg, _ViaSite)
_BOOK = ("metal", "via_pads", "cuts", "persistent")
_LOCAL = ("required_states", "baselines", "landings", "grouped_pin_keys",
          "portals_by_net", "net_nodes", "net_node_pins", "committed_groups",
          "group_contributions")
_EXACT = ("_attempt_cuts", "_attempt_metal", "_attempt_idx", "_edge_fail",
          "_route_failure_details", "_hard_rejections", "_hard_rejection_samples",
          "_astar_watchdog_details", "_access_watchdog_details", "_route_watchdog_details",
          "_selected_access_pair", "_last_accepted_access_states",
          "_last_accepted_terminal_probes", "_last_candidate_terminal_probes",
          "_bounds", "_rail_by_net")
_PREFIX = ("_active_", "_group_route_", "_search_forbidden_")


def _pin_sources(value, memo, seen):
    """Keep physical terminals and immutable routing primitives source-owned."""
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, _PINNED):
        memo[identity] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _pin_sources(key, memo, seen)
            _pin_sources(item, memo, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _pin_sources(item, memo, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            _pin_sources(getattr(value, f.name), memo, seen)


def _names(engine):
    return set(_EXACT) | {name for name in vars(engine)
                          if name.startswith(_PREFIX)}


def capture(book, engine, local):
    """Capture all group-mutable state while leaving budget and clocks live."""
    targets = {("book", name): getattr(book, name) for name in _BOOK}
    targets.update({("local", name): local[name] for name in _LOCAL})
    names = _names(engine)
    raw = {"book": {name: getattr(book, name) for name in _BOOK},
           "local": {name: local[name] for name in _LOCAL},
           "engine": {name: getattr(engine, name) for name in names
                      if hasattr(engine, name)}}
    memo = {}
    _pin_sources(raw, memo, set())
    snapshot = deepcopy(raw, memo)
    return targets, snapshot


def _canonical(value, active):
    """Encode full structural values, failing closed on unsupported objects."""
    if isinstance(value, Enum):
        return ["enum", type(value).__module__, type(value).__qualname__, value.name]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite checkpoint float")
        return ["float", value.hex()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic checkpoint state")
    active.add(identity)
    try:
        if isinstance(value, dict):
            items = [[_canonical(k, active), _canonical(v, active)] for k, v in value.items()]
            items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
            return ["dict", items]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [_canonical(v, active) for v in value]]
        if isinstance(value, (set, frozenset)):
            items = [_canonical(v, active) for v in value]
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
            return [type(value).__name__, items]
        kind = [type(value).__module__, type(value).__qualname__]
        if isinstance(value, _SpatialIndex):
            # Canonicalize via logical (buckets, cell) state so digest() stays
            # 100% bit-identical whether C++20 NativeSpatialIndex is active or not.
            return [
                "slots",
                kind,
                [
                    ["buckets", _canonical(value.buckets, active)],
                    ["cell", _canonical(value.cell, active)],
                ],
            ]
        if is_dataclass(value) and not isinstance(value, type):
            return ["dataclass", kind, [[f.name, _canonical(getattr(value, f.name), active)] for f in fields(value)]]
        if hasattr(value, "__dict__"):
            return ["object", kind, _canonical(vars(value), active)]
        slots = getattr(type(value), "__slots__", None)
        if slots:
            if isinstance(slots, str):
                slots = (slots,)
            return ["slots", kind, [[s, _canonical(getattr(value, s), active)] for s in sorted(slots) if hasattr(value, s)]]
        raise TypeError(f"unsupported checkpoint value {type(value)!r}")
    finally:
        active.remove(identity)


def digest(checkpoint):
    """Hash restorable content without incorporating IDs or live counters."""
    payload = _canonical(checkpoint[1], set())
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _refill(target, source, memo):
    if isinstance(target, dict) and isinstance(source, dict):
        target.clear()
        for key, value in source.items():
            target[deepcopy(key, memo)] = deepcopy(value, memo)
    elif isinstance(target, list) and isinstance(source, list):
        target[:] = [deepcopy(value, memo) for value in source]
    elif isinstance(target, set) and isinstance(source, set):
        target.clear()
        target.update(deepcopy(value, memo) for value in source)
    else:
        raise TypeError("checkpoint root must be a matching mutable container")


def restore(checkpoint, book, engine, local):
    """Restore a repeatable checkpoint, then refresh committed-world indexes."""
    targets, snapshot = checkpoint
    memo = {}
    _pin_sources(snapshot, memo, set())
    for (scope, name), target in targets.items():
        memo[id(snapshot[scope][name])] = target
    for (scope, name), target in targets.items():
        _refill(target, snapshot[scope][name], memo)
        if scope == "book":
            setattr(book, name, target)
        else:
            local[name] = target
    for name in _names(engine):
        if name not in snapshot["engine"] and hasattr(engine, name):
            delattr(engine, name)
    for name, value in snapshot["engine"].items():
        setattr(engine, name, deepcopy(value, memo))
    engine._book = book
    engine._rebuild_metal_index()
    engine._wide_index_metal_index = None
