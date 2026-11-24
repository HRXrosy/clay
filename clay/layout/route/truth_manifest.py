"""Immutable identity binding for the exact GDS consumed by signoff gates."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional


class ArtifactDigestMismatch(RuntimeError):
    """A signoff verdict was requested for bytes other than the bound artifact."""


def sha256_path(path: Path) -> str:
    """Return the SHA-256 of a file without loading the whole artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_nonempty_path(path: Path, *, what: str) -> str:
    candidate = Path(path).resolve()
    try:
        size = candidate.stat().st_size
    except OSError as exc:
        raise ArtifactDigestMismatch(
            f"signoff {what} is missing or unreadable: {candidate}"
        ) from exc
    if size <= 0:
        raise ArtifactDigestMismatch(
            f"signoff {what} must be a non-empty file: {candidate}"
        )
    return sha256_path(candidate)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha256_text(text: str) -> str:
    """Return the SHA-256 of the exact UTF-8 bytes written for generated input."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_ring_nm(points: Any) -> tuple[tuple[int, int], ...]:
    try:
        ring = tuple(
            (
                int(round(float(point[0]) * 1000.0)),
                int(round(float(point[1]) * 1000.0)),
            )
            for point in points
        )
    except Exception as exc:
        raise ValueError("final placement geometry has a malformed polygon") from exc
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError("final placement geometry has a degenerate polygon")
    candidates = []
    for values in (ring, tuple(reversed(ring))):
        candidates.extend(
            values[index:] + values[:index] for index in range(len(values))
        )
    return min(candidates)


def final_placement_artwork_sha256(
    placement_um: Mapping[str, Any], geometry: Mapping[str, Any]
) -> str:
    """Canonical digest of the actual placed stamps consumed by ``route_cell``."""
    if not isinstance(placement_um, Mapping) or not isinstance(geometry, Mapping):
        raise TypeError("final placement and geometry must be mappings")
    if not placement_um or not geometry:
        raise ValueError("final placement and geometry must be non-empty")

    def _keys_by_name(values: Mapping[str, Any], *, what: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for raw_name in values:
            name = str(raw_name)
            if not name or name in result:
                raise ValueError(f"final {what} must name unique non-empty stamps")
            result[name] = raw_name
        return result

    def _coordinate_nm(raw_xy: Any, *, name: str) -> tuple[int, int]:
        try:
            x_um, y_um = raw_xy
            x = float(x_um)
            y = float(y_um)
        except Exception as exc:
            raise ValueError(f"final placement {name!r} has malformed coordinates") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"final placement {name!r} has non-finite coordinates")
        return int(round(x * 1000.0)), int(round(y * 1000.0))

    placement_keys = _keys_by_name(placement_um, what="placement")
    geometry_keys = _keys_by_name(geometry, what="geometry")
    placement_names = set(placement_keys)
    geometry_names = set(geometry_keys)
    missing_physical = geometry_names - placement_names
    if missing_physical:
        raise ValueError(
            "final placement lacks physical geometry stamps: "
            + ", ".join(sorted(missing_physical))
        )

    composite_members: dict[str, tuple[str, ...]] = {}
    alias_owners: dict[str, list[str]] = {}
    for name in sorted(geometry_names):
        stamp = geometry[geometry_keys[name]]
        raw_members = getattr(stamp, "composite_members", ())
        if raw_members is None:
            raw_members = ()
        if (
            type(raw_members) is not tuple
            or len(raw_members) != len(set(raw_members))
            or any(type(member) is not str or not member for member in raw_members)
        ):
            raise ValueError(
                f"final placement {name!r} has malformed composite_members provenance"
            )
        members = tuple(raw_members)
        composite_members[name] = members
        for member in members:
            alias_owners.setdefault(member, []).append(name)

    aliases = placement_names - geometry_names
    alias_payload = []
    for name in sorted(aliases):
        owners = alias_owners.get(name, [])
        if not owners:
            raise ValueError(
                f"final placement alias {name!r} is not declared by a composite stamp"
            )
        if len(owners) != 1:
            raise ValueError(
                f"final placement alias {name!r} is declared by multiple composite stamps: "
                + ", ".join(sorted(owners))
            )
        x, y = _coordinate_nm(placement_um[placement_keys[name]], name=name)
        alias_payload.append((name, owners[0], x, y))

    payload = []
    for name in sorted(geometry_names):
        raw_geometry_name = geometry_keys[name]
        raw_placement_name = placement_keys[name]
        x, y = _coordinate_nm(placement_um[raw_placement_name], name=name)
        stamp = geometry[raw_geometry_name]
        layers = []
        raw_layers = getattr(stamp, "polys_by_layer", None)
        if not isinstance(raw_layers, Mapping):
            raise TypeError(f"final placement {name!r} lacks typed stamp geometry")
        for raw_layer, rings in sorted(
            raw_layers.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
        ):
            key = (int(raw_layer[0]), int(raw_layer[1]))
            canonical = tuple(sorted(_canonical_ring_nm(ring) for ring in rings))
            if canonical:
                layers.append((key, canonical))
        payload.append(
            (
                name,
                x,
                y,
                str(getattr(stamp, "source", "") or ""),
                composite_members[name],
                tuple(layers),
            )
        )
    return _canonical_sha256(
        {
            "schema": "clay.final-placement-artwork.v2",
            "stamps": payload,
            "composite_aliases": alias_payload,
        }
    )


@dataclass(frozen=True)
class TruthManifest:
    """Hashes that identify one signoff run's immutable inputs.

    The four supporting hashes are explicit provenance, while
    :meth:`bind_verdict_gds` intentionally depends only on the exact-GDS fields.
    Adding or consuming input identities therefore cannot weaken artifact binding.
    """

    exact_gds_path: Path
    exact_gds_sha256: str
    source_path: Optional[Path] = None
    source_sha256: Optional[str] = None
    source_netlist_path: Optional[Path] = None
    source_netlist_sha256: Optional[str] = None
    reference_netlist_path: Optional[Path] = None
    netlist_sha256: Optional[str] = None
    pdk_sha256: Optional[str] = None
    lvs_deck_path: Optional[Path] = None
    lvs_deck_sha256: Optional[str] = None
    drc_deck_path: Optional[Path] = None
    drc_deck_sha256: Optional[str] = None
    full_extraction_lvsdb_path: Optional[Path] = None
    full_extraction_lvsdb_sha256: Optional[str] = None
    final_placement_sha256: Optional[str] = None
    final_placement_certificate_sha256: Optional[str] = None
    row_final_placement_sha256: Optional[str] = None
    route_manifest_sha256: Optional[str] = None

    @classmethod
    def from_artifact(
        cls,
        path: Path,
        *,
        source_path: Optional[Path] = None,
        source_netlist_path: Optional[Path] = None,
        netlist_path: Optional[Path] = None,
        netlist_text: Optional[str] = None,
        pdk_identity: Any = None,
        lvs_deck_path: Optional[Path] = None,
        lvs_deck_text: Optional[str] = None,
        drc_deck_path: Optional[Path] = None,
        drc_deck_text: Optional[str] = None,
        full_extraction_lvsdb_path: Optional[Path] = None,
        final_placement: Any = None,
        final_placement_sha256: Optional[str] = None,
        final_placement_certificate_sha256: Optional[str] = None,
        row_final_placement_sha256: Optional[str] = None,
        route_manifest_sha256: Optional[str] = None,
    ) -> "TruthManifest":
        if netlist_path is not None and netlist_text is not None:
            raise ValueError("provide netlist_path or netlist_text, not both")
        if lvs_deck_path is not None and lvs_deck_text is not None:
            raise ValueError("provide lvs_deck_path or lvs_deck_text, not both")
        if drc_deck_path is not None and drc_deck_text is not None:
            raise ValueError("provide drc_deck_path or drc_deck_text, not both")
        if final_placement is not None and final_placement_sha256 is not None:
            raise ValueError(
                "provide final_placement or final_placement_sha256, not both"
            )
        exact_path = Path(path).resolve()
        source = Path(source_path).resolve() if source_path is not None else None
        source_netlist = (
            Path(source_netlist_path).resolve()
            if source_netlist_path is not None
            else None
        )
        reference = Path(netlist_path).resolve() if netlist_path is not None else None
        lvs_deck = Path(lvs_deck_path).resolve() if lvs_deck_path is not None else None
        drc_deck = Path(drc_deck_path).resolve() if drc_deck_path is not None else None
        lvsdb = (
            Path(full_extraction_lvsdb_path).resolve()
            if full_extraction_lvsdb_path is not None
            else None
        )
        for name, digest in (
            ("row_final_placement_sha256", row_final_placement_sha256),
            ("route_manifest_sha256", route_manifest_sha256),
            (
                "final_placement_certificate_sha256",
                final_placement_certificate_sha256,
            ),
            ("final_placement_sha256", final_placement_sha256),
        ):
            if digest is not None and not _is_sha256(digest):
                raise ValueError(f"{name} must be one lowercase SHA-256")
        return cls(
            exact_gds_path=exact_path,
            exact_gds_sha256=sha256_path(exact_path),
            source_path=source,
            source_sha256=(
                _sha256_nonempty_path(source, what="source")
                if source is not None
                else None
            ),
            source_netlist_path=source_netlist,
            source_netlist_sha256=(
                _sha256_nonempty_path(source_netlist, what="source netlist")
                if source_netlist is not None
                else None
            ),
            reference_netlist_path=reference,
            netlist_sha256=(
                _sha256_nonempty_path(reference, what="reference netlist")
                if reference is not None
                else sha256_text(netlist_text) if netlist_text is not None else None
            ),
            pdk_sha256=(_canonical_sha256(pdk_identity)
                        if pdk_identity is not None else None),
            lvs_deck_path=lvs_deck,
            lvs_deck_sha256=(
                _sha256_nonempty_path(lvs_deck, what="LVS deck")
                if lvs_deck is not None
                else sha256_text(lvs_deck_text) if lvs_deck_text is not None else None
            ),
            drc_deck_path=drc_deck,
            drc_deck_sha256=(
                _sha256_nonempty_path(drc_deck, what="DRC deck")
                if drc_deck is not None
                else sha256_text(drc_deck_text) if drc_deck_text is not None else None
            ),
            full_extraction_lvsdb_path=lvsdb,
            full_extraction_lvsdb_sha256=(
                _sha256_nonempty_path(lvsdb, what="full-extraction LVS database")
                if lvsdb is not None
                else None
            ),
            final_placement_sha256=(
                final_placement_sha256
                if final_placement_sha256 is not None
                else _canonical_sha256(final_placement)
                if final_placement is not None
                else None
            ),
            final_placement_certificate_sha256=(
                final_placement_certificate_sha256
            ),
            row_final_placement_sha256=row_final_placement_sha256,
            route_manifest_sha256=route_manifest_sha256,
        )

    def _validate_bound_path(
        self,
        path: Optional[Path],
        expected_sha256: Optional[str],
        *,
        what: str,
    ) -> Path:
        if path is None or not _is_sha256(expected_sha256):
            raise ArtifactDigestMismatch(f"signoff provenance lacks {what} bytes")
        candidate = Path(path).resolve()
        actual = _sha256_nonempty_path(candidate, what=what)
        if actual != expected_sha256:
            raise ArtifactDigestMismatch(
                f"signoff {what} bytes changed: expected sha256={expected_sha256}, "
                f"got {candidate} sha256={actual}"
            )
        return candidate

    def bind_full_extraction_outputs(
        self,
        *,
        reference_netlist_path: Path,
        lvs_deck_path: Path,
        lvsdb_path: Path,
        drc_deck_path: Path,
    ) -> "TruthManifest":
        """Return a manifest rebound to the files the real oracles consumed.

        Pre-emission generated text hashes are treated as expectations.  The
        files written for the real KLayout process must replay those hashes;
        merely reporting a plausible path is never sufficient.
        """
        self.validate_current_artifact()
        reference = Path(reference_netlist_path).resolve()
        lvs_deck = Path(lvs_deck_path).resolve()
        lvsdb = Path(lvsdb_path).resolve()
        drc_deck = Path(drc_deck_path).resolve()
        observed = {
            "reference netlist": _sha256_nonempty_path(
                reference, what="reference netlist"
            ),
            "LVS deck": _sha256_nonempty_path(lvs_deck, what="LVS deck"),
            "DRC deck": _sha256_nonempty_path(drc_deck, what="DRC deck"),
        }
        expected = {
            "reference netlist": self.netlist_sha256,
            "LVS deck": self.lvs_deck_sha256,
            "DRC deck": self.drc_deck_sha256,
        }
        for what, actual in observed.items():
            if actual != expected[what]:
                raise ArtifactDigestMismatch(
                    f"signoff {what} is not the pre-bound input: "
                    f"expected sha256={expected[what]}, got sha256={actual}"
                )
        return replace(
            self,
            reference_netlist_path=reference,
            netlist_sha256=observed["reference netlist"],
            lvs_deck_path=lvs_deck,
            lvs_deck_sha256=observed["LVS deck"],
            drc_deck_path=drc_deck,
            drc_deck_sha256=observed["DRC deck"],
            full_extraction_lvsdb_path=lvsdb,
            full_extraction_lvsdb_sha256=_sha256_nonempty_path(
                lvsdb, what="full-extraction LVS database"
            ),
        )

    def validate_signoff_provenance(self) -> None:
        """Re-read every file-backed input of a completed exact signoff run."""
        self.validate_current_artifact()
        for name in (
            "source_sha256",
            "source_netlist_sha256",
            "netlist_sha256",
            "pdk_sha256",
            "lvs_deck_sha256",
            "drc_deck_sha256",
            "full_extraction_lvsdb_sha256",
            "final_placement_sha256",
            "final_placement_certificate_sha256",
            "row_final_placement_sha256",
            "route_manifest_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ArtifactDigestMismatch(
                    f"signoff provenance lacks canonical {name}"
                )
        self.validate_source_provenance()
        self._validate_bound_path(
            self.reference_netlist_path,
            self.netlist_sha256,
            what="reference netlist",
        )
        self._validate_bound_path(
            self.lvs_deck_path, self.lvs_deck_sha256, what="LVS deck"
        )
        self._validate_bound_path(
            self.drc_deck_path, self.drc_deck_sha256, what="DRC deck"
        )
        self._validate_bound_path(
            self.full_extraction_lvsdb_path,
            self.full_extraction_lvsdb_sha256,
            what="full-extraction LVS database",
        )

    def validate_source_provenance(self) -> None:
        """Re-read the declarative Clay source and independently supplied circuit."""
        self._validate_bound_path(
            self.source_path, self.source_sha256, what="source"
        )
        self._validate_bound_path(
            self.source_netlist_path,
            self.source_netlist_sha256,
            what="source netlist",
        )

    def bind_verdict_gds(self, path: Path) -> Path:
        """Return ``path`` only when it is byte-identical to the emitted GDS."""
        candidate = Path(path).resolve()
        candidate_sha256 = sha256_path(candidate)
        if candidate_sha256 != self.exact_gds_sha256:
            raise ArtifactDigestMismatch(
                "signoff verdict does not bind the exact emitted GDS: "
                f"expected {self.exact_gds_path} sha256={self.exact_gds_sha256}, "
                f"got {candidate} sha256={candidate_sha256}"
            )
        return candidate

    def bind_claimed_sha256(self, claimed_sha256: str, *, what: str) -> str:
        """Bind a typed evidence claim to this manifest's exact GDS bytes.

        This is the in-memory dual of :meth:`bind_verdict_gds`: reports which
        already carry an artifact digest can be joined without re-reading or
        re-emitting geometry, while a forged/cross-run digest fails closed.
        """
        claimed = str(claimed_sha256 or "")
        if claimed != self.exact_gds_sha256:
            raise ArtifactDigestMismatch(
                f"{what} does not bind the exact emitted GDS: "
                f"expected sha256={self.exact_gds_sha256}, got sha256={claimed}"
            )
        return claimed

    def bind_pdk_identity(self, pdk_identity: Any) -> str:
        """Require current PDK source identity to match emission-time identity."""

        measured = _canonical_sha256(pdk_identity)
        if not _is_sha256(self.pdk_sha256) or measured != self.pdk_sha256:
            raise ArtifactDigestMismatch(
                "signoff PDK identity differs from the exact emitted artifact: "
                f"expected={self.pdk_sha256}, got={measured}"
            )
        return measured

    def with_pdk_identity(
        self,
        pdk_identity: Any,
        *,
        route_manifest_sha256: Optional[str] = None,
    ) -> "TruthManifest":
        """Add emission-time PDK identity without dropping existing lineage."""

        self.validate_current_artifact()
        if route_manifest_sha256 is not None and not _is_sha256(
            route_manifest_sha256
        ):
            raise ValueError("route_manifest_sha256 must be one lowercase SHA-256")
        return replace(
            self,
            pdk_sha256=_canonical_sha256(pdk_identity),
            route_manifest_sha256=(
                self.route_manifest_sha256
                if route_manifest_sha256 is None
                else route_manifest_sha256
            ),
        )

    def validate_current_artifact(self) -> Path:
        """Re-read the bound path and reject post-construction byte mutation."""
        return self.bind_verdict_gds(self.exact_gds_path)


__all__ = [
    "ArtifactDigestMismatch",
    "TruthManifest",
    "final_placement_artwork_sha256",
    "sha256_path",
    "sha256_text",
]
