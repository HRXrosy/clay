"""Test-isolation helpers for real-ngspice and multi-PDK pytest runs.

When multiple test modules that run real ``ngspice_con`` simulations or switch
PDKs execute in the same pytest process, mutating process-global state can leak
from an earlier module into later ones (for example, pointing ``CLAY_PDK_ID``
and ``CLAY_PDK_PATH`` at ``gf180mcu`` causes subsequent ``sky130`` tests to
resolve against the wrong PDK and produce degenerate / ``nan`` outputs).

A second, compounding vector is the module-level config cache in
``clay.config`` (``_GLOBAL_CONFIG``): once a stale-PDK config is resolved and
cached, subsequent ``get_config()`` calls keep returning it even after the
environment is restored.

Mechanism
---------
Snapshot the Clay PDK/simulator environment keys (and cached-config presence)
before a module or block that may mutate them, and restore the snapshot
afterward, invalidating the ``clay.config`` cache so the next resolution
re-reads the restored environment. This changes no circuit result: each test
still sees the exact configuration its module requested while preventing
cross-module leakage.

A defense-in-depth :func:`reap_stray_ngspice` helper is also provided to detect
(and optionally terminate) any ``ngspice_con`` child that a timed-out or
crashed run might leave behind. It is non-destructive by default (pure
observation); reaping requires an explicit ``kill=True``.

Nothing here is imported by production code paths; it is wired only into the
test harness (an ``autouse`` fixture in ``tests/conftest.py``). Importing this
module has no side effects.
"""

from __future__ import annotations

import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


# The full set of environment keys that steer Clay's PDK / simulator
# resolution. A test that mutates ANY of these can poison a later test;
# we snapshot/restore the whole set so the order of these keys never
# matters across files in one process.
CLAY_PDK_ENV_KEYS: Tuple[str, ...] = (
    "CLAY_PDK_ID",
    "CLAY_PDK_PATH",
    "CLAY_NGSPICE_BIN",
    "CLAY_WORK_DIR",
    "CLAY_CONFIG",
    # PDK-specific path overrides that adapters honour.
    "CLAY_GF180MCU_PATH",
    "CLAY_SKY130_PATH",
)

# The Windows silent-batch binary basename and the POSIX one. Used by the
# stray-process reaper to know what image name to look for.
_NGSPICE_IMAGE_WIN = "ngspice_con.exe"
_NGSPICE_IMAGE_POSIX = "ngspice"


# ─── env / config state snapshot ───────────────────────────────────────


@dataclass
class ClayStateSnapshot:
    """A captured copy of the Clay PDK/sim env keys + cached-config flag.

    ``env`` maps each key in :data:`CLAY_PDK_ENV_KEYS` to its value at
    snapshot time, or ``None`` if the key was unset. ``had_cached_config``
    records whether ``clay.config._GLOBAL_CONFIG`` was populated, purely
    for diagnostics; restore always re-reads from the restored env.
    """

    env: Dict[str, Optional[str]] = field(default_factory=dict)
    had_cached_config: bool = False


def snapshot_clay_state() -> ClayStateSnapshot:
    """Capture the current Clay PDK/sim env + cached-config presence."""
    env: Dict[str, Optional[str]] = {
        key: os.environ.get(key) for key in CLAY_PDK_ENV_KEYS
    }
    had_cached = False
    try:
        from clay import config as _cfg

        had_cached = _cfg._GLOBAL_CONFIG is not None
    except Exception:  # noqa: BLE001 - config import must never break tests
        had_cached = False
    return ClayStateSnapshot(env=env, had_cached_config=had_cached)


def reset_config_cache() -> None:
    """Invalidate ``clay.config``'s module-level config singleton.

    The next ``get_config()`` then re-resolves from the (restored)
    environment. No-op if ``clay.config`` cannot be imported.
    """
    try:
        from clay import config as _cfg

        _cfg._GLOBAL_CONFIG = None
    except Exception:  # noqa: BLE001
        pass


def restore_clay_state(snap: ClayStateSnapshot) -> None:
    """Restore env keys to the snapshot and invalidate the config cache.

    Keys that were unset at snapshot time are removed (not set to the
    empty string), so ``ClayConfig.auto`` sees exactly the pre-test
    environment.
    """
    for key, value in snap.env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reset_config_cache()


def reestablish_module_pdk_env(module) -> Optional[str]:
    """Re-run a test module's own PDK-env setup convention, if it has one.

    Many real-ngspice test modules establish their PDK by calling a
    private ``_ensure_<pdk>_env()`` helper — but only as a side effect of a
    ``@pytest.mark.skipif`` marker, which pytest evaluates ONCE at
    collection time. With cross-module env isolation in place, that
    collection-time mutation is rolled back, so by the time the module's
    tests actually run the env may no longer point at the module's PDK.

    This helper restores idempotency by re-invoking the module's own
    setup at module-fixture setup time, regardless of run order. It looks
    for a zero-argument callable named ``_ensure_<something>_env`` on the
    module (the established convention in this repo) and calls it.

    Args:
        module: the test module object (``request.module`` in a fixture).

    Returns:
        The name of the setup function that was invoked, or ``None`` if
        the module declares no such convention (then the module relies on
        the ambient / default config, which the baseline already provides).
    """
    if module is None:
        return None
    for attr in dir(module):
        if attr.startswith("_ensure_") and attr.endswith("_env"):
            fn = getattr(module, attr, None)
            if callable(fn):
                try:
                    fn()
                    return attr
                except Exception:  # noqa: BLE001 - never break setup
                    return None
    return None


@contextmanager
def isolated_clay_state() -> Iterator[ClayStateSnapshot]:
    """Context manager: snapshot Clay PDK/sim state, restore on exit.

    Anything the body writes into the Clay PDK env keys (or into the
    config cache) is rolled back when the block exits, so a test that
    repoints the PDK cannot leak that choice into the next test.
    """
    snap = snapshot_clay_state()
    try:
        yield snap
    finally:
        restore_clay_state(snap)


# ─── stray ngspice process reaper (defense in depth) ────────────────────


def _ngspice_image_name() -> str:
    return _NGSPICE_IMAGE_WIN if platform.system() == "Windows" else _NGSPICE_IMAGE_POSIX


def list_stray_ngspice() -> List[int]:
    """Return PIDs of currently-running ``ngspice`` / ``ngspice_con`` procs.

    Pure observation: never terminates anything. Returns an empty list on
    any failure (the tool being absent, a permissions error, etc.) so a
    caller can treat "could not enumerate" the same as "none found".
    """
    image = _ngspice_image_name()
    pids: List[int] = []
    try:
        if platform.system() == "Windows":
            # CSV output: "image","pid","session","sess#","mem"
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                text=True,
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if not line or line.startswith('"INFO:'):
                    # tasklist prints 'INFO: No tasks...' when none match.
                    continue
                cols = [c.strip('"') for c in line.split('","')]
                if len(cols) >= 2 and cols[0].lower() == image.lower():
                    try:
                        pids.append(int(cols[1]))
                    except ValueError:
                        continue
        else:
            out = subprocess.run(
                ["pgrep", "-x", image],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                text=True,
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
    except Exception:  # noqa: BLE001
        return []
    return pids


def reap_stray_ngspice(
    *, kill: bool = False, exclude_pids: Optional[List[int]] = None
) -> List[int]:
    """Find (and optionally terminate) stray ngspice processes.

    Args:
        kill: when False (default) this is a pure observation — it returns
            the list of stray PIDs and terminates nothing. When True it
            attempts to terminate each stray PID (``taskkill /F`` on
            Windows, ``SIGKILL`` on POSIX).
        exclude_pids: PIDs never to touch even when ``kill=True`` (e.g. a
            child the caller intends to keep alive).

    Returns:
        The list of stray PIDs that were found (and, when ``kill=True``,
        targeted for termination). Best-effort: failures to kill are
        swallowed so end-of-test cleanup never raises.
    """
    exclude = set(exclude_pids or ())
    pids = [p for p in list_stray_ngspice() if p not in exclude]
    if not kill:
        return pids
    for pid in pids:
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
            else:
                os.kill(pid, 9)
        except Exception:  # noqa: BLE001
            continue
    return pids
