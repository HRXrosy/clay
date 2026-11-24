"""Source declarations for routing changes to an authenticated imported layout.

The imported layout is an explicit input, not a generated primitive library.
Only observed, net-owned route resources may change; ordinary placement cannot
consume these declarations. Distances on this surface are integer nanometres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import re


def _digest(value: object, field: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"imported routing {field} requires a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ImportedLayoutIntent:
    manifest: str
    sha256: str

    def __post_init__(self):
        if (type(self.manifest) is not str or not self.manifest.strip()
                or self.manifest != self.manifest.strip()
                or any(c in self.manifest for c in "\r\n\0")):
            raise ValueError("imported layout requires a canonical manifest path")
        _digest(self.sha256, "manifest identity")

    def canonical_data(self):
        return asdict(self)


@dataclass(frozen=True)
class RouteTailIntent:
    net: str
    resource: str
    trim_nm: int

    def __post_init__(self):
        if type(self.net) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.net) is None:
            raise ValueError("route tail net requires a canonical source identifier")
        if type(self.resource) is not str or re.fullmatch(r"tail_[0-9a-f]{64}", self.resource) is None:
            raise ValueError("route resource requires its canonical observed tail_ identity")
        if type(self.trim_nm) is not int or self.trim_nm < 0:
            raise ValueError("route tail trim_nm requires a nonnegative integer")

    def canonical_data(self):
        return asdict(self)


def canonical_imported_routes(imported, tails):
    if imported is not None:
        if type(imported) is not ImportedLayoutIntent:
            raise TypeError("imported layout requires its typed source declaration")
        ImportedLayoutIntent(**imported.canonical_data())
    if type(tails) is not tuple:
        raise TypeError("route tail inventory must be an exact tuple")
    resources = set()
    for tail in tails:
        if type(tail) is not RouteTailIntent:
            raise TypeError("route tail inventory requires typed declarations")
        RouteTailIntent(**tail.canonical_data())
        if tail.resource in resources:
            raise ValueError("one imported route resource may be declared only once")
        resources.add(tail.resource)
    if tails and imported is None:
        raise ValueError("route_tail requires an authenticated imported_layout")
    return imported, tails


def route_tail_from_declaration(decl):
    attrs = decl.attributes or {}
    if decl.kind != "route_tail" or len(decl.nets) != 1 or set(attrs) != {"resource", "trim_nm"}:
        raise ValueError("route_tail requires one net, resource and trim_nm")
    try:
        amount = Decimal(str(attrs["trim_nm"]))
    except InvalidOperation as exc:
        raise ValueError("route_tail trim_nm is not an exact integer") from exc
    if not amount.is_finite() or amount != amount.to_integral_value() or amount < 0:
        raise ValueError("route_tail trim_nm must be a nonnegative exact integer")
    return RouteTailIntent(decl.nets[0], attrs["resource"], int(amount))


def imported_routes_from_declarations(declarations):
    imported, tails = None, []
    for decl in declarations:
        attrs = decl.attributes or {}
        if decl.kind == "imported_layout":
            if imported is not None or decl.nets or set(attrs) != {"manifest", "sha256"}:
                raise ValueError("exactly one imported_layout with manifest and sha256 is allowed")
            imported = ImportedLayoutIntent(**attrs)
        elif decl.kind == "route_tail":
            tails.append(route_tail_from_declaration(decl))
    return canonical_imported_routes(imported, tuple(tails))


def merge_imported_routes(left_imported, left_tails, right_imported, right_tails):
    canonical_imported_routes(left_imported, left_tails)
    canonical_imported_routes(right_imported, right_tails)
    if left_imported is not None and right_imported is not None and left_imported != right_imported:
        raise ValueError("cannot merge different imported layout authorities")
    tails = tuple(dict.fromkeys(left_tails + right_tails))
    return canonical_imported_routes(right_imported or left_imported, tails)
