"""PDK-bound normalization of reference segmented-resistor PCell intent."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Literal, Mapping

from clay.flow.reference_evidence import canonical_topology_from_spice
from clay.frontend.reference_pcell_reconstruction import (
    ReferenceComponent,
    ReferencePCellReconstructionError,
    _decimal_spice_quantum,
    _decimal_spice_value,
    _fixed_decimal_context,
    _topology_member_name,
)
from clay.pdk import get_pdk


_CDF_DEFAULT_NAMES = frozenset(
    {
        "connection",
        "cktl",
        "cktw",
        "r",
        "segl",
        "segr",
        "segments",
        "segw",
    }
)
_MAX_SEGMENTS = 4096
_CDF_MODELS = frozenset({"npolyf_u", "ppolyf_u"})


@dataclass(frozen=True, slots=True)
class ReferenceResistorSegmentIntent:
    """One coordinate-free resistor PCell declaration with source authority."""

    source_instance: str
    def_instance: str
    model: str
    pdk_id: str
    count: int
    connection: Literal["series", "parallel"]
    width_um: Decimal
    length_um: Decimal
    segment_resistance_ohms: Decimal
    segment_resistance_quantum_ohms: Decimal
    value_quantum_ohms: Decimal
    cdf_relative_path: str
    cdf_sha256: str

    def __post_init__(self) -> None:
        self.assert_authoritative_schema()

    def assert_authoritative_schema(self) -> None:
        """Replay the exact immutable schema after possible object mutation."""

        for label, value in (
            ("source instance", self.source_instance),
            ("DEF instance", self.def_instance),
            ("model", self.model),
            ("PDK", self.pdk_id),
            ("CDF path", self.cdf_relative_path),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise TypeError(f"reference resistor {label} is not exact")
        if type(self.count) is not int or not (1 <= self.count <= _MAX_SEGMENTS):
            raise ValueError("reference resistor segment count is outside the bound")
        if self.connection not in {"series", "parallel"}:
            raise ValueError("reference resistor connection is unsupported")
        for label, value in (
            ("width", self.width_um),
            ("length", self.length_um),
            ("segment resistance", self.segment_resistance_ohms),
            ("value quantum", self.value_quantum_ohms),
        ):
            if type(value) is not Decimal or not value.is_finite() or value <= 0:
                raise ValueError(f"reference resistor {label} must be exact and positive")
        if (
            type(self.segment_resistance_quantum_ohms) is not Decimal
            or not self.segment_resistance_quantum_ohms.is_finite()
            or self.segment_resistance_quantum_ohms < 0
        ):
            raise ValueError(
                "reference resistor segment quantum must be exact and non-negative"
            )
        if (
            type(self.cdf_sha256) is not str
            or len(self.cdf_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.cdf_sha256)
        ):
            raise ValueError("reference resistor CDF digest is not canonical")
        cdf_path = PurePosixPath(self.cdf_relative_path)
        if (
            "\\" in self.cdf_relative_path
            or cdf_path.is_absolute()
            or PureWindowsPath(self.cdf_relative_path).drive
            or ".." in cdf_path.parts
            or str(cdf_path) != self.cdf_relative_path
        ):
            raise ValueError("reference resistor CDF path is not canonical")
        if self.model not in _CDF_MODELS:
            raise ValueError("reference resistor model lacks CDF authority")

    def render(self) -> str:
        """Render the canonical Clay annotation consumed by the materializer."""

        self.assert_authoritative_schema()
        segment_quantum = (
            "segment_resistance_quantum_ohms="
            f"{_decimal_source(self.segment_resistance_quantum_ohms)}, "
            if self.segment_resistance_quantum_ohms > 0
            else ""
        )
        return (
            f"  {self.source_instance}: @resistor_segments("
            f"pdk={self.pdk_id}, count={self.count}, "
            f"connection={self.connection}, "
            f"width_um={_decimal_source(self.width_um)}, "
            f"length_um={_decimal_source(self.length_um)}, "
            "segment_resistance_ohms="
            f"{_decimal_source(self.segment_resistance_ohms)}, "
            f"{segment_quantum}"
            f"value_quantum_ohms={_decimal_source(self.value_quantum_ohms)});"
        )


@dataclass(frozen=True, slots=True)
class _CapturedCdf:
    root: Path
    relative_path: str
    sha256: str
    payload: bytes
    signature: tuple[int, int, int, int]


def _decimal_source(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _capture_bound_cdf(
    adapter,
    authority: Mapping[str, object],
    *,
    model: str,
) -> _CapturedCdf:
    if model not in _CDF_MODELS:
        raise ReferencePCellReconstructionError(
            f"segmented-resistor model {model!r} lacks CDF authority"
        )
    row = authority.get(model)
    if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
        raise ReferencePCellReconstructionError(
            f"segmented-resistor model {model!r} CDF authority is incomplete"
        )
    raw_path = row.get("path")
    expected_sha256 = row.get("sha256")
    if type(raw_path) is not str or not raw_path:
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF path is missing"
        )
    pure_path = PurePosixPath(raw_path)
    if (
        "\\" in raw_path
        or pure_path.is_absolute()
        or PureWindowsPath(raw_path).drive
        or ".." in pure_path.parts
        or str(pure_path) != raw_path
    ):
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF path is not canonical"
        )
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF digest is not canonical"
        )
    root = adapter.root()
    if root is None:
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF install root is unavailable"
        )
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / Path(*pure_path.parts)).resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF path is unavailable"
        ) from exc
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF escapes the install root"
        ) from exc
    if not candidate.is_file():
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF is not a regular file"
        )

    def signature() -> tuple[int, int, int, int]:
        stat = candidate.stat()
        return (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(getattr(stat, "st_ino", 0)),
        )

    before = signature()
    payload = candidate.read_bytes()
    after = signature()
    if before != after or not payload:
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF changed while reading"
        )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ReferencePCellReconstructionError(
            "segmented-resistor PCell CDF digest differs"
        )
    return _CapturedCdf(
        root=resolved_root,
        relative_path=raw_path,
        sha256=observed_sha256,
        payload=payload,
        signature=after,
    )


def _cdf_defaults(payload: bytes) -> dict[str, str]:
    """Parse the exact default fields from the captured OA CDF record stream."""

    if type(payload) is not bytes or not payload:
        raise TypeError("reference resistor CDF payload must be exact bytes")
    records = payload.split(b"(storeDefault ")[1:]
    defaults: dict[str, str] = {}
    for record in records:
        name_match = re.search(rb'\bname "([^"]+)"', record)
        if name_match is None:
            continue
        try:
            name = name_match.group(1).decode("ascii").casefold()
        except UnicodeDecodeError as exc:
            raise ReferencePCellReconstructionError(
                "segmented-resistor CDF parameter name is not ASCII"
            ) from exc
        if name not in _CDF_DEFAULT_NAMES:
            continue
        value_matches = tuple(
            re.finditer(rb'\bdefValue (?:"([^"]*)"|([^\s()]+))', record)
        )
        if len(value_matches) != 1:
            raise ReferencePCellReconstructionError(
                f"segmented-resistor CDF {name} default is ambiguous"
            )
        raw_value = value_matches[0].group(1) or value_matches[0].group(2)
        try:
            value = raw_value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ReferencePCellReconstructionError(
                f"segmented-resistor CDF {name} default is not ASCII"
            ) from exc
        if not value or name in defaults:
            raise ReferencePCellReconstructionError(
                f"segmented-resistor CDF {name} default is not unique"
            )
        defaults[name] = value
    if set(defaults) != _CDF_DEFAULT_NAMES:
        missing = tuple(sorted(_CDF_DEFAULT_NAMES - set(defaults)))
        raise ReferencePCellReconstructionError(
            f"segmented-resistor CDF default inventory differs: missing={missing}"
        )
    return defaults


def _positive_integer(raw: str, *, label: str) -> int:
    try:
        value = Decimal(raw)
    except (DecimalException, ValueError) as exc:
        raise ReferencePCellReconstructionError(
            f"{label} is not an exact positive integer"
        ) from exc
    if (
        not value.is_finite()
        or value <= 0
        or value != value.to_integral_value()
        or value > _MAX_SEGMENTS
    ):
        raise ReferencePCellReconstructionError(
            f"{label} is not an exact positive integer"
        )
    return int(value)


def _instance_order(intent: ReferenceResistorSegmentIntent) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Za-z_]+)(\d+)", intent.source_instance)
    if match is None:
        return intent.source_instance.casefold(), -1, intent.source_instance
    return match.group(1).casefold(), int(match.group(2)), intent.source_instance


def _derive_one(
    component: ReferenceComponent,
    *,
    source_instance: str,
    pdk_id: str,
    defaults: Mapping[str, str],
    cdf: _CapturedCdf,
) -> ReferenceResistorSegmentIntent:
    parameters = dict(component.parameters)
    count = _positive_integer(
        parameters.get("segments", defaults["segments"]),
        label=f"{component.instance} segments",
    )
    connection = parameters.get("connection", defaults["connection"]).casefold()
    if connection not in {"series", "parallel"}:
        raise ReferencePCellReconstructionError(
            f"{component.instance} has unsupported segment connection"
        )
    count_parameter = "ssegs" if connection == "series" else "psegs"
    if count_parameter in parameters and _positive_integer(
        parameters[count_parameter],
        label=f"{component.instance} {count_parameter}",
    ) != count:
        raise ReferencePCellReconstructionError(
            f"{component.instance} segment count declarations differ"
        )

    width_raw = parameters.get("segw", defaults["segw"])
    width_m = _decimal_spice_value(
        width_raw, label=f"{component.instance} segment width"
    )
    segment_resistance_raw = parameters.get("segr", defaults["segr"])
    segment_resistance = _decimal_spice_value(
        segment_resistance_raw,
        label=f"{component.instance} segment resistance",
    )
    length_raw = parameters.get("segl", defaults["segl"])
    length_m = _decimal_spice_value(
        length_raw,
        label=f"{component.instance} segment length",
    )

    if "cktl" in parameters:
        circuit_length = _decimal_spice_value(
            parameters["cktl"],
            label=f"{component.instance} circuit length",
        )
        expected_length = (
            length_m * Decimal(count) if connection == "series" else length_m
        )
        if circuit_length != expected_length:
            raise ReferencePCellReconstructionError(
                f"{component.instance} circuit length differs from segment topology"
            )
    if "cktw" in parameters:
        circuit_width = _decimal_spice_value(
            parameters["cktw"],
            label=f"{component.instance} circuit width",
        )
        expected_width = (
            width_m if connection == "series" else width_m * Decimal(count)
        )
        if circuit_width != expected_width:
            raise ReferencePCellReconstructionError(
                f"{component.instance} circuit width differs from segment topology"
            )

    if "r" not in parameters:
        raise ReferencePCellReconstructionError(
            f"{component.instance} lacks required schematic resistance"
        )
    value_raw = parameters["r"]
    value = _decimal_spice_value(value_raw, label=f"{component.instance} resistance")
    value_quantum = _decimal_spice_quantum(
        value_raw, label=f"{component.instance} resistance"
    )
    segment_quantum = _decimal_spice_quantum(
        segment_resistance_raw,
        label=f"{component.instance} segment resistance",
    )
    with localcontext(_fixed_decimal_context()):
        if connection == "series":
            equivalent = segment_resistance * Decimal(count)
            segment_uncertainty = segment_quantum * Decimal(count) / Decimal(2)
        else:
            equivalent = segment_resistance / Decimal(count)
            segment_uncertainty = segment_quantum / Decimal(count) / Decimal(2)
        total_uncertainty = value_quantum / Decimal(2) + segment_uncertainty
    if abs(equivalent - value) > total_uncertainty:
        raise ReferencePCellReconstructionError(
            f"{component.instance} segment equivalent differs from schematic value"
        )
    reported_segment_quantum = (
        segment_quantum
        if abs(equivalent - value) > value_quantum / Decimal(2)
        else Decimal(0)
    )

    with localcontext(_fixed_decimal_context()):
        width_um = width_m * Decimal("1e6")
        length_um = length_m * Decimal("1e6")
    return ReferenceResistorSegmentIntent(
        source_instance=source_instance,
        def_instance=component.instance,
        model=component.master,
        pdk_id=pdk_id,
        count=count,
        connection=connection,
        width_um=width_um,
        length_um=length_um,
        segment_resistance_ohms=segment_resistance,
        segment_resistance_quantum_ohms=reported_segment_quantum,
        value_quantum_ohms=value_quantum,
        cdf_relative_path=cdf.relative_path,
        cdf_sha256=cdf.sha256,
    )


def derive_reference_resistor_segment_intents(
    components: tuple[ReferenceComponent, ...],
    *,
    topology_text: str,
    pdk_id: str,
) -> tuple[ReferenceResistorSegmentIntent, ...]:
    """Normalize every reference resistor using DEF fields and bound CDF defaults."""

    if type(components) is not tuple:
        raise TypeError("reference component inventory requires an exact immutable tuple")
    if any(type(component) is not ReferenceComponent for component in components):
        raise TypeError("reference component inventory contains a noncanonical member")
    component_keys = tuple(component.instance.casefold() for component in components)
    if (
        component_keys != tuple(sorted(component_keys))
        or len(set(component_keys)) != len(component_keys)
    ):
        raise TypeError(
            "reference component inventory requires canonical order and uniqueness"
        )
    if type(topology_text) is not str:
        raise TypeError("reference topology must be exact text")
    if type(pdk_id) is not str or not pdk_id:
        raise TypeError("reference resistor normalization requires a PDK ID")
    adapter = get_pdk(pdk_id)
    raw_extra = getattr(adapter, "extra", None)
    if not isinstance(raw_extra, Mapping):
        raise ReferencePCellReconstructionError(
            "PDK lacks segmented-resistor parameter authority"
        )
    authority = raw_extra.get("reference_resistor_cdf_authority")
    if not isinstance(authority, Mapping):
        raise ReferencePCellReconstructionError(
            "PDK lacks segmented-resistor parameter authority"
        )
    resistor_models = tuple(
        sorted(
            {
                component.master
                for component in components
                if component.device_class == "resistor"
            }
        )
    )
    if not resistor_models:
        raise ReferencePCellReconstructionError(
            "reference resistor obligation inventory is empty"
        )
    authorities = {
        model: (
            captured := _capture_bound_cdf(adapter, authority, model=model),
            _cdf_defaults(captured.payload),
        )
        for model in resistor_models
    }
    topology = canonical_topology_from_spice(topology_text)
    topology_by_name = topology.by_name()
    topology_names = set(topology_by_name)

    intents = []
    for component in components:
        component.assert_authoritative()
        if component.device_class != "resistor":
            continue
        source_instance = _topology_member_name(component, topology_names)
        topology_member = topology_by_name[source_instance]
        if topology_member.kind != "resistor":
            raise ReferencePCellReconstructionError(
                f"{component.instance} topology member is not a resistor"
            )
        canonical_source_instance = topology_member.instance
        cdf, defaults = authorities[component.master]
        intents.append(
            _derive_one(
                component,
                source_instance=canonical_source_instance,
                pdk_id=pdk_id,
                defaults=defaults,
                cdf=cdf,
            )
        )
    topology_resistors = {
        name
        for name, member in topology_by_name.items()
        if member.kind == "resistor"
    }
    normalized_resistors = {
        intent.source_instance.casefold() for intent in intents
    }
    if normalized_resistors != topology_resistors:
        raise ReferencePCellReconstructionError(
            "reference topology resistor denominator differs from schematic DEF"
        )
    for model, (cdf, _defaults) in authorities.items():
        confirmed = _capture_bound_cdf(adapter, authority, model=model)
        if (
            confirmed.root != cdf.root
            or confirmed.relative_path != cdf.relative_path
            or confirmed.sha256 != cdf.sha256
            or confirmed.signature != cdf.signature
            or confirmed.payload != cdf.payload
        ):
            raise ReferencePCellReconstructionError(
                "segmented-resistor PCell CDF authority changed during replay"
            )
    return tuple(sorted(intents, key=_instance_order))


def render_reference_resistor_segment_annotations(
    components: tuple[ReferenceComponent, ...],
    *,
    topology_text: str,
    pdk_id: str,
) -> str:
    """Render the complete nonempty canonical annotation inventory."""

    intents = derive_reference_resistor_segment_intents(
        components,
        topology_text=topology_text,
        pdk_id=pdk_id,
    )
    return "\n".join(intent.render() for intent in intents)


__all__ = [
    "ReferenceResistorSegmentIntent",
    "derive_reference_resistor_segment_intents",
    "render_reference_resistor_segment_annotations",
]
