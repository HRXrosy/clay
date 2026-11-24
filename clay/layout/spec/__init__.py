"""Pluggable spec-pack subsystem (SP1).

A spec pack is a loadable, named collection of design RULES. Each rule pairs
the (metric, effect, object, hardness) taxonomy tuple with a topological
detector, a zero-coordinate emitter, and a named independent checker. Packs are
opt-in: nothing is loaded by default, so behaviour stays byte-identical until a
pack is explicitly registered and selected.

This package is pure data structures + a module-level registry. It does not
touch any pipeline; SP2 wires emitters into the QSR relation network.
"""
from clay.layout.spec.packs import (
    SpecRule,
    SpecPack,
    register_pack,
    load_pack,
    list_packs,
    packs_for_category,
    pack_as_dict,
)

__all__ = [
    "SpecRule",
    "SpecPack",
    "register_pack",
    "load_pack",
    "list_packs",
    "packs_for_category",
    "pack_as_dict",
]
