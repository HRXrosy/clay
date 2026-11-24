"""Small, circuit-independent provenance vocabulary for physical planning.

These references identify the declaration or rule that caused a physical
object to exist.  They deliberately carry no coordinates and no circuit role;
later certificates can therefore name their sources without teaching a placer
or router about a particular design.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProvenanceKind(str, Enum):
    USER = "user"
    TOPOLOGY = "topology"
    PDK = "pdk"
    REFERENCE_ORACLE = "reference_oracle"


@dataclass(frozen=True)
class DeclarationRef:
    kind: ProvenanceKind
    identifier: str


__all__ = ["DeclarationRef", "ProvenanceKind"]
