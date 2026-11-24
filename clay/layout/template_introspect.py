"""Introspection for cell template factories.

Exposes factory signatures, default param values, and required vs optional
kwargs via Python's ``inspect`` module. Pure stdlib, no Rust mirror.

Public API:
- ``TemplateParam`` / ``TemplateSignature`` (frozen dataclasses)
- ``introspect_template(name)``
- ``list_all_templates()``
- ``list_required_params(name)``
- ``list_optional_params(name)``
- ``get_template_docstring(name)``
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cell_templates import (
    build_diff_pair,
    build_current_mirror,
    build_cascoded_current_mirror,
    build_5t_ota,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
    build_bandgap_reference,
)


# Map short template name -> factory callable.
# Keep ordering deterministic so list_all_templates() is stable.
_TEMPLATES: Dict[str, Callable[..., Any]] = {
    "diff_pair": build_diff_pair,
    "current_mirror": build_current_mirror,
    "cascoded_current_mirror": build_cascoded_current_mirror,
    "five_t_ota": build_5t_ota,
    "folded_cascode": build_folded_cascode,
    "telescopic_cascode": build_telescopic_cascode,
    "two_stage_miller": build_two_stage_miller,
    "bandgap_reference": build_bandgap_reference,
}


# Map inspect.Parameter.kind -> human-readable string used in TemplateParam.
_KIND_NAMES: Dict[Any, str] = {
    inspect.Parameter.POSITIONAL_ONLY: "positional",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional",
    inspect.Parameter.VAR_POSITIONAL: "var_positional",
    inspect.Parameter.KEYWORD_ONLY: "keyword_only",
    inspect.Parameter.VAR_KEYWORD: "var_keyword",
}


@dataclass(frozen=True)
class TemplateParam:
    """A single parameter of a cell-template factory.

    Attributes:
        name: parameter name as declared on the factory.
        kind: one of "positional", "keyword_only", "var_positional",
            "var_keyword".
        default: default value, or ``inspect.Parameter.empty`` when required.
        annotation: type annotation, or ``inspect.Parameter.empty`` when none.
    """

    name: str
    kind: str
    default: Any
    annotation: Any


@dataclass(frozen=True)
class TemplateSignature:
    """Full signature of a cell-template factory.

    Attributes:
        template_name: short name (key into ``list_all_templates()``).
        params: ordered list of ``TemplateParam`` in declaration order.
        docstring: factory ``__doc__`` (may be ``None``).
    """

    template_name: str
    params: List[TemplateParam] = field(default_factory=list)
    docstring: Optional[str] = None


def _resolve(template_name: str) -> Callable[..., Any]:
    """Resolve a short template name to its factory, or raise ValueError."""
    try:
        return _TEMPLATES[template_name]
    except KeyError:
        known = ", ".join(sorted(_TEMPLATES))
        raise ValueError(
            f"Unknown template {template_name!r}. Known templates: {known}"
        ) from None


def introspect_template(template_name: str) -> TemplateSignature:
    """Introspect a cell-template factory by short name.

    Uses :func:`inspect.signature` and converts each
    :class:`inspect.Parameter` to a :class:`TemplateParam`.

    Raises:
        ValueError: if ``template_name`` is unknown.
    """
    factory = _resolve(template_name)
    sig = inspect.signature(factory)
    params: List[TemplateParam] = []
    for p in sig.parameters.values():
        kind = _KIND_NAMES.get(p.kind, str(p.kind))
        params.append(
            TemplateParam(
                name=p.name,
                kind=kind,
                default=p.default,
                annotation=p.annotation,
            )
        )
    return TemplateSignature(
        template_name=template_name,
        params=params,
        docstring=inspect.getdoc(factory),
    )


def list_all_templates() -> List[str]:
    """Return the 8 known template short names in declaration order."""
    return list(_TEMPLATES.keys())


def list_required_params(template_name: str) -> List[str]:
    """Return names of parameters without a default value (required)."""
    sig = introspect_template(template_name)
    return [
        p.name
        for p in sig.params
        if p.default is inspect.Parameter.empty
        and p.kind not in ("var_positional", "var_keyword")
    ]


def list_optional_params(template_name: str) -> List[str]:
    """Return names of parameters that have a default value (optional)."""
    sig = introspect_template(template_name)
    return [
        p.name
        for p in sig.params
        if p.default is not inspect.Parameter.empty
    ]


def get_template_docstring(template_name: str) -> Optional[str]:
    """Return the factory docstring, or ``None`` if absent."""
    return introspect_template(template_name).docstring
