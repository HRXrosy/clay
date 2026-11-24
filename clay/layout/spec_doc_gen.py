"""Spec doc generator for cell-template factories.

Combines Phase 52 introspection (``template_introspect``) with Phase 14
benchmark metadata (``template_benchmark``) to auto-produce a markdown
spec document for each cell-template factory.

Pure Python (stdlib only). Does not modify either upstream module.

Public API:
- ``generate_template_spec_doc(template_name)``
- ``generate_all_template_docs()``
- ``write_template_docs(output_dir)``
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Union

from .template_introspect import (
    TemplateParam,
    TemplateSignature,
    introspect_template,
    list_all_templates,
)


def _factory_name(template_name: str) -> str:
    """Return the public factory name (``build_<template_name>``).

    Mirrors the naming convention used in ``clay.layout.cell_templates``,
    where ``diff_pair`` is built by ``build_diff_pair`` etc. The factory
    name shows up as the markdown H1 heading.
    """
    return f"build_{template_name}"


def _format_default(default: Any) -> str:
    """Format a parameter default for the markdown table.

    ``inspect.Parameter.empty`` renders as an em dash, which is the
    convention used elsewhere in Clay docs for "no default / required".
    Other defaults render via ``repr`` so strings keep quotes and
    sentinel values stay unambiguous.
    """
    if default is inspect.Parameter.empty:
        return "—"
    return repr(default)


def _is_required(param: TemplateParam) -> bool:
    """A parameter is required iff it has no default and is not var-args."""
    if param.default is not inspect.Parameter.empty:
        return False
    if param.kind in ("var_positional", "var_keyword"):
        return False
    return True


def _render_params_table(sig: TemplateSignature) -> str:
    """Render the parameter table portion of the spec doc."""
    header = (
        "| Name | Kind | Default | Required |\n"
        "| --- | --- | --- | --- |"
    )
    if not sig.params:
        # Empty-signature factories (none today, but bandgap_reference is
        # close) still need a well-formed table for downstream parsers.
        return header + "\n| _(no parameters)_ | — | — | — |"

    rows = []
    for p in sig.params:
        required = "yes" if _is_required(p) else "no"
        rows.append(
            f"| `{p.name}` | {p.kind} | {_format_default(p.default)} | {required} |"
        )
    return header + "\n" + "\n".join(rows)


def generate_template_spec_doc(template_name: str) -> str:
    """Generate the markdown spec doc for a single cell-template factory.

    The output starts with ``# build_<name>`` as an H1 heading, followed
    by the factory docstring (or a placeholder if absent), then a
    ``## Parameters`` table built from
    :func:`clay.layout.template_introspect.introspect_template`.

    Raises:
        ValueError: if ``template_name`` is not a known template
            (propagated from ``introspect_template``).
    """
    sig = introspect_template(template_name)
    factory = _factory_name(template_name)

    docstring = sig.docstring if sig.docstring else "_No docstring available._"
    table = _render_params_table(sig)

    parts = [
        f"# {factory}",
        "",
        docstring,
        "",
        "## Parameters",
        "",
        table,
        "",
    ]
    return "\n".join(parts)


def generate_all_template_docs() -> Dict[str, str]:
    """Generate spec docs for every known cell template.

    Returns a dict keyed by template short-name (matching
    :func:`list_all_templates`) with markdown spec docs as values. The
    iteration order matches ``list_all_templates()`` so the output is
    deterministic.
    """
    return {name: generate_template_spec_doc(name) for name in list_all_templates()}


def write_template_docs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Write each template's spec doc to ``<output_dir>/<name>.md``.

    Creates ``output_dir`` (and parents) if absent. Returns a dict
    mapping template short-name to the absolute output ``Path``, which
    makes it easy for callers (or tests) to verify the artifacts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = generate_all_template_docs()
    written: Dict[str, Path] = {}
    for name, markdown in docs.items():
        path = out / f"{name}.md"
        path.write_text(markdown, encoding="utf-8")
        written[name] = path.resolve()
    return written
