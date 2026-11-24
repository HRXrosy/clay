"""
clay/_rust_bridge_core.py - Shared thin core for the Rust-delegate bridges
(R159 Agent C, BL-159C; M-RUST-ALGOCORE strangler-fig support layer).

Every Rust port adds a Python bridge file that repeats the same
boilerplate. Two families exist:

  * maturin ``clay_ir`` kernels (``clay/_ir_bridge_*_rs.py``) - each
    repeats a ``try: import clay_ir ... except ImportError`` import-gate,
    a set of ``_HAS_<name> = hasattr(clay_ir, "try_rust_...")`` capability
    probes, and a ``<module>_rust_enabled()`` gate that ANDs them.

  * ``clay_layout`` CLI kernels (``clay/layout/_rs_bridge_*.py``) - each
    repeats binary discovery, the ``subprocess.run`` invocation with
    timeout / OSError / non-zero-returncode handling, JSON marshalling,
    and a ``rust_available()`` probe.

This module extracts both shared shapes so each bridge shrinks to a thin
declaration (kernel name, fallback fn, signature). It introduces ZERO
behaviour change: every helper here is byte-for-byte equivalent to the
inline boilerplate it replaces.

The maturin side imports ``clay_ir`` exactly once here and re-uses the
loaded handle, so the cost of the (possibly failing) import is paid a
single time per process rather than once per bridge module.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


# ──────────────────────────────────────────────────────────────────────
# maturin ``clay_ir`` gate core (for clay/_ir_bridge_*_rs.py)
# ──────────────────────────────────────────────────────────────────────


def load_clay_ir() -> Optional[Any]:
    """Best-effort import of the maturin ``clay_ir`` extension module.

    Returns the imported module, or ``None`` when it is not built /
    importable. Mirrors the ``try: import clay_ir ... except
    ImportError: clay_ir = None`` prelude every ``_ir_bridge_*_rs.py``
    bridge used to repeat verbatim.
    """
    try:
        import clay_ir  # type: ignore[import-untyped]

        return clay_ir
    except ImportError:
        return None


# Module-level shared handle: imported once for the whole process. A bridge
# can read this directly when it needs to call ``clay_ir.try_rust_*`` from
# its public delegate, instead of re-running the import.
clay_ir: Optional[Any] = load_clay_ir()


def probe_entries(
    module: Optional[Any],
    names: Sequence[str],
) -> Dict[str, bool]:
    """Return ``{name: hasattr(module, name)}`` for each PyO3 entry name.

    When ``module`` is ``None`` (extension unbuilt) every flag is
    ``False``. Mirrors the per-bridge block of ``_HAS_<x> =
    hasattr(clay_ir, "try_rust_...")`` assignments byte-equal.
    """
    if module is None:
        return {name: False for name in names}
    return {name: hasattr(module, name) for name in names}


def make_enabled_gate(
    names: Sequence[str],
    module: Optional[Any] = None,
) -> Tuple[Optional[Any], Callable[[], bool]]:
    """Build the ``(module, enabled_gate)`` pair for a maturin bridge.

    ``names`` is the full list of ``try_rust_*_ir`` PyO3 entry names the
    bridge depends on. ``module`` defaults to the shared :data:`clay_ir`
    handle but may be overridden (e.g. in tests). The returned gate is a
    zero-arg callable returning ``True`` iff EVERY entry is present on the
    module — exactly the ``all(_HAS_*)`` semantics each bridge's
    ``<module>_rust_enabled()`` used to compute inline.

    The probe is evaluated eagerly (once, at gate-construction time) so
    the gate is a cheap constant lookup, matching the old module-level
    ``_HAS_*`` flags which were computed at import time.
    """
    mod = module if module is not None else clay_ir
    flags = probe_entries(mod, names)
    enabled = all(flags.values())

    def _enabled() -> bool:
        return enabled

    return mod, _enabled


# ──────────────────────────────────────────────────────────────────────
# clay_layout CLI subprocess core (for clay/layout/_rs_bridge_*.py)
# ──────────────────────────────────────────────────────────────────────
#
# These helpers are parameterised by an injected ``find_binary`` callable
# and an ``error_cls`` so the core does not import ``clay.layout`` (which
# would create an import cycle: clay.layout.* imports this module). Each
# layout bridge already imports ``_find_binary`` / ``BinaryGdsSubprocessError``
# from ``.binary_gds_subprocess`` and now passes them through.


def resolve_layout_binary(
    find_binary: Callable[[], Optional[Path]],
    binary_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the clay_layout CLI binary path.

    Returns the explicit ``binary_path`` if it exists, else the result of
    the injected ``find_binary`` discovery. ``None`` when no usable binary
    is present. Mirrors the ``bp = binary_path if ... else _find_binary()``
    line each bridge repeated.
    """
    bp = binary_path if binary_path is not None else find_binary()
    if bp is None:
        return None
    bp = Path(bp)
    return bp if bp.exists() else None


def layout_rust_available(
    find_binary: Callable[[], Optional[Path]],
    binary_path: Optional[Path] = None,
) -> bool:
    """True if the clay_layout CLI binary is PRESENT on disk.

    Honest scope: this checks existence of the resolved binary path only. It does
    NOT verify the binary's source, version, freshness vs the Rust source, or that
    it is actually executable -- a stale/incompatible binary still reports True.
    A real version/executability probe (``<bin> --version``) is a deliberate
    follow-up; callers needing provenance must check it separately.

    Byte-equal replacement for the ``rust_available`` body each layout
    bridge repeated.
    """
    return resolve_layout_binary(find_binary, binary_path) is not None


def invoke_layout_kernel(
    subcommand: str,
    payload: Mapping[str, Any],
    *,
    find_binary: Callable[[], Optional[Path]],
    error_cls: type,
    binary_path: Optional[Path] = None,
    timeout_sec: float = 60.0,
    missing_binary_hint: str = "",
) -> dict:
    """Invoke ``<clay_layout_gds_bin> <subcommand>`` (JSON in / JSON out).

    Builds the binary command, serialises ``payload`` to JSON on stdin,
    runs the subprocess with ``capture_output=True, text=True``, maps a
    missing binary / timeout / OSError / non-zero return code to
    ``error_cls`` with the same messages the bridges used, and returns the
    parsed JSON dict from stdout.

    This is the byte-equal extraction of the ``_rust_compute`` body that
    every ``_rs_bridge_*.py`` repeated; the per-kernel JSON payload shape
    is the only specific the caller supplies.
    """
    bin_path = resolve_layout_binary(find_binary, binary_path)
    if bin_path is None:
        hint = (
            missing_binary_hint
            # crates/clay_layout is its OWN isolated workspace (NOT a member of the
            # root Cargo.toml, which holds only clay_parser + clay_ir). Build it
            # from inside its own manifest, not with `-p clay_layout` from root.
            or "  (cd crates/clay_layout && cargo build --release)"
        )
        raise error_cls(
            "Rust CLI binary `clay_layout_gds_bin` not found. Build it first:\n"
            + hint
        )

    body = json.dumps(payload)
    cmd = [str(bin_path), subcommand]
    try:
        proc = subprocess.run(
            cmd,
            input=body,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise error_cls(
            f"{subcommand} subcommand timed out after {timeout_sec}s"
        ) from e
    except OSError as e:
        raise error_cls(
            f"failed to spawn {subcommand} subcommand: {e}"
        ) from e

    if proc.returncode != 0:
        raise error_cls(
            f"{subcommand} exited {proc.returncode}\n"
            f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
        )
    return json.loads(proc.stdout)


__all__ = [
    "clay_ir",
    "load_clay_ir",
    "probe_entries",
    "make_enabled_gate",
    "resolve_layout_binary",
    "layout_rust_available",
    "invoke_layout_kernel",
]
