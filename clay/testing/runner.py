"""Unified high-speed, parallel, OpenBLAS-safe pytest runner for Clay (`clay test`).

Solves the four structural pytest slowness traps on Windows workstations:
1. Prevents Anaconda plugin-autoload ABI crashes (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`)
   while still explicitly enabling parallel execution via `-p xdist -n <N> --dist worksteal`.
2. Caps BLAS/OpenMP threads per worker (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
   `MKL_NUM_THREADS=1`) so N workers never spawn N×32 oversubscribed threads.
3. Provides clean, explicit tiers:
   - ``fast`` (default): git blast-radius changed-file tests in parallel (seconds)
   - ``core``: curated C++20 + fastpath + sizing + flow + CLI + continuity core gate (~25s)
   - ``regression``: parallel continuity regression via ``scripts/check_engineering_continuity.py --regression``
   - ``all``: full hermetic parallel gate + serial ``perf_timing`` lane
   - ``perf_timing``: serial wall-clock timing probe lane
4. Avoids Windows command-line length overflow via automatic pytest ``@argfile`` staging.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
SAFE_COMMAND_LIMIT = 24_000

FAST_GATE_MARKER = (
    "not signoff and not slow and not requires_simulator and not requires_llm"
)

#: High-signal core verification cohort covering C++20 PnR, fastpath parity,
#: sizing-to-layout, flow contracts, CLI single-entry, and continuity preflight.
CORE_FAST_TARGETS: tuple[str, ...] = (
    "tests/test_single_entry_and_sizing_cpp_robustness.py",
    "tests/test_cpp_modular_architecture.py",
    "tests/test_cpp_engine_phase0_parity.py",
    "tests/test_cpp_engine_phase23_parity.py",
    "tests/test_cpp_engine_phase34_parity.py",
    "tests/test_fast_test_runner.py",
    "tests/test_engineering_continuity.py",
    "tests/test_flow_spec_resolution.py",
    "tests/test_flow_kpi_source.py",
    "tests/test_physical_support.py",
    "tests/test_live_candidate_diagnostic_caller.py",
    "tests/test_collision_geometry_session.py",
    "tests/test_core_fastpath_parity.py",
    "tests/test_cpp_placement_kernels.py",
    "tests/test_cpp_pathfinder_kernels.py",
    "tests/test_cpp_gridless_kernels.py",
    "tests/test_bandgap_auto_e2e.py",
)

#: Native C++20 differential & architecture verification suite (`--tier cpp`).
CPP_FAST_TARGETS: tuple[str, ...] = (
    "tests/test_cpp_modular_architecture.py",
    "tests/test_cpp_engine_phase0_parity.py",
    "tests/test_cpp_engine_phase23_parity.py",
    "tests/test_cpp_engine_phase34_parity.py",
    "tests/test_cpp_placement_kernels.py",
    "tests/test_cpp_pathfinder_kernels.py",
    "tests/test_cpp_gridless_kernels.py",
    "tests/test_core_fastpath_parity.py",
)


def xdist_available() -> bool:
    """True when ``pytest-xdist`` is importable in the active interpreter."""
    return importlib.util.find_spec("xdist") is not None


def resolve_workers(requested: str | int = "auto", *, max_workers: int = 16) -> int:
    """Resolve parallel xdist worker count, capping to avoid RAM/BLAS thrashing."""
    if not xdist_available():
        return 1
    if isinstance(requested, int):
        return max(1, requested)
    raw = str(requested).strip().lower()
    if raw in {"0", "1", "serial", "none", "false"}:
        return 1
    if raw == "auto":
        cpus = os.cpu_count() or 2
        return max(1, min(max_workers, cpus // 2))
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def safe_pytest_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a clean, autoload-disabled, OpenBLAS-capped env for parallel pytest."""
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.setdefault("CLAY_PDK_ID", "gf180mcu")
    gf_root = REPO_ROOT / "pdks" / "gf180mcu"
    if gf_root.is_dir():
        env.setdefault("CLAY_GF180MCU_PATH", str(gf_root))
        env.setdefault("CLAY_PDK_PATH", str(gf_root))
    return env


def build_pytest_command(
    targets: Sequence[str] = (),
    *,
    workers: str | int = "auto",
    marker: str | None = None,
    durations: int | None = None,
    keyword: str | None = None,
    last_failed: bool = False,
    extra_args: Sequence[str] = (),
    python_exe: str | None = None,
) -> list[str]:
    """Construct a parallel-safe pytest command line."""
    n = resolve_workers(workers)
    cmd = [python_exe or sys.executable, "-m", "pytest", "-q"]
    if n > 1:
        cmd.extend(["-p", "xdist", "-n", str(n), "--dist", "worksteal"])
    if marker:
        cmd.extend(["-m", marker])
    if last_failed:
        cmd.append("--lf")
    if keyword:
        cmd.extend(["-k", keyword])
    if durations is not None and durations >= 0:
        cmd.append(f"--durations={int(durations)}")
    cmd.extend(targets)
    cmd.extend(extra_args)
    return cmd


def _git_lines(*args: str, repo_root: Path = REPO_ROOT) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def changed_python_files(base: str | None = None, *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return changed `.py` files under `clay/`, `scripts/`, `tools/`, or `tests/`."""
    if base:
        files = _git_lines("diff", "--name-only", f"{base}...HEAD", repo_root=repo_root)
        files += _git_lines("diff", "--name-only", repo_root=repo_root)
        files += _git_lines("diff", "--name-only", "--cached", repo_root=repo_root)
    else:
        files = _git_lines("diff", "--name-only", repo_root=repo_root)
        files += _git_lines("diff", "--name-only", "--cached", repo_root=repo_root)
        files += _git_lines("ls-files", "--others", "--exclude-standard", repo_root=repo_root)
    seen: list[str] = []
    for rel in files:
        if rel not in seen and rel.endswith(".py") and rel.startswith(("clay/", "tests/", "scripts/", "tools/")):
            seen.append(rel)
    return seen


def map_changed_to_tests(
    changed: Iterable[str],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], list[str]]:
    """Map changed Python paths to candidate test files via single-pass scan."""
    changed_list = list(changed)
    selected: list[str] = []
    uncovered: list[str] = []
    patterns: dict[str, tuple[re.Pattern[str], re.Pattern[str] | None]] = {}

    for rel in changed_list:
        p = repo_root / rel
        if rel.startswith("tests/") and p.name.startswith("test_"):
            if p.is_file() and rel not in selected:
                selected.append(rel)
            continue
        parts = rel[:-3].split("/")
        full = ".".join(parts)
        parent = ".".join(parts[:-1])
        stem = parts[-1]
        if stem == "__init__":
            full = parent
            parent = ".".join(parts[:-2])
            stem = parts[-2] if len(parts) >= 2 else ""
        pat_full = re.compile(rf"\b{re.escape(full)}\b")
        pat_from = (
            re.compile(rf"from\s+{re.escape(parent)}\s+import\b[^\n]*\b{re.escape(stem)}\b")
            if parent and stem
            else None
        )
        patterns[rel] = (pat_full, pat_from)

    if patterns:
        hits_by_src: dict[str, list[str]] = {rel: [] for rel in patterns}
        tests_dir = repo_root / "tests"
        for tf in sorted(tests_dir.glob("test_*.py")):
            try:
                txt = tf.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            tf_rel = tf.relative_to(repo_root).as_posix()
            for src_rel, (pat_full, pat_from) in patterns.items():
                if pat_full.search(txt) or (pat_from is not None and pat_from.search(txt)):
                    hits_by_src[src_rel].append(tf_rel)
        for src_rel, matches in hits_by_src.items():
            if not matches:
                uncovered.append(src_rel)
            else:
                for m in matches:
                    if m not in selected:
                        selected.append(m)
    return selected, uncovered


def _exec_pytest_with_argfile(
    base_cmd: list[str],
    targets: Sequence[str],
    extra_args: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path = REPO_ROOT,
) -> int:
    full_cmd = [*base_cmd, *targets, *extra_args]
    size = sum(len(a) + 3 for a in full_cmd)
    if size <= SAFE_COMMAND_LIMIT:
        return subprocess.run(full_cmd, cwd=str(cwd), env=env).returncode
    with tempfile.TemporaryDirectory(prefix="clay-test-") as td:
        argfile = Path(td) / "targets.txt"
        argfile.write_text("".join(f"{t}\n" for t in targets), encoding="utf-8")
        short_cmd = [*base_cmd, f"@{argfile}", *extra_args]
        print(f"[clay test] command length {size} chars -> using @argfile ({len(targets)} targets)")
        return subprocess.run(short_cmd, cwd=str(cwd), env=env).returncode


def run_clay_tests(
    *,
    tier: str = "fast",
    targets: Sequence[str] = (),
    workers: str | int = "auto",
    base: str | None = None,
    last_failed: bool = False,
    keyword: str | None = None,
    durations: int | None = None,
    list_only: bool = False,
    extra_args: Sequence[str] = (),
    python_exe: str | None = None,
) -> int:
    """Execute the selected Clay test tier with safe parallel xdist configuration."""
    py = python_exe or sys.executable
    env = safe_pytest_env()
    t0 = time.perf_counter()

    if tier == "regression":
        n = resolve_workers(workers)
        cmd = [py, "scripts/check_engineering_continuity.py", "--regression", "--workers", str(n)]
        if list_only:
            print(" ".join(cmd))
            return 0
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode

    if targets:
        existing = [t for t in targets]
        if list_only:
            for item in existing:
                print(item)
            return 0
        base_cmd = build_pytest_command(
            workers=workers,
            durations=durations,
            keyword=keyword,
            last_failed=last_failed,
            python_exe=py,
        )
        rc = _exec_pytest_with_argfile(base_cmd, existing, extra_args, env=env)
        print(f"[clay test] done in {time.perf_counter() - t0:.2f}s (exit={rc})")
        return rc

    if last_failed:
        cmd = build_pytest_command(
            workers=workers,
            marker=FAST_GATE_MARKER,
            durations=durations,
            keyword=keyword,
            last_failed=True,
            extra_args=extra_args,
            python_exe=py,
        )
        if list_only:
            print(" ".join(cmd))
            return 0
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
        print(f"[clay test] --lf done in {time.perf_counter() - t0:.2f}s (exit={rc})")
        return rc

    if tier == "cpp":
        cpp_targets = [t for t in CPP_FAST_TARGETS if (REPO_ROOT / t).is_file()]
        build_cmd = [py, "cpp/clay_pnr/build.py", "--target", "tests", "--test"]
        if list_only:
            print("CPP_NATIVE:", " ".join(build_cmd))
            for t in cpp_targets:
                print(t)
            return 0
        print("[clay test] tier=cpp step 1/2: pure C++20 native unit tests")
        rc_cpp = subprocess.run(build_cmd, cwd=str(REPO_ROOT), env=env).returncode
        if rc_cpp != 0:
            print(f"[clay test] tier=cpp FAILED in native C++20 tests (exit={rc_cpp})")
            return rc_cpp
        base_cmd = build_pytest_command(
            workers=workers,
            durations=durations,
            keyword=keyword,
            python_exe=py,
        )
        print(f"[clay test] tier=cpp step 2/2: Python<->C++20 parity ({len(cpp_targets)} files, workers={resolve_workers(workers)})")
        rc_py = _exec_pytest_with_argfile(base_cmd, cpp_targets, extra_args, env=env)
        print(f"[clay test] tier=cpp done in {time.perf_counter() - t0:.2f}s (exit={rc_py})")
        return rc_py

    if tier == "core":
        core_targets = [t for t in CORE_FAST_TARGETS if (REPO_ROOT / t).is_file()]
        if list_only:
            for t in core_targets:
                print(t)
            return 0
        base_cmd = build_pytest_command(
            workers=workers,
            durations=durations,
            keyword=keyword,
            python_exe=py,
        )
        print(f"[clay test] tier=core ({len(core_targets)} files, workers={resolve_workers(workers)})")
        rc = _exec_pytest_with_argfile(base_cmd, core_targets, extra_args, env=env)
        print(f"[clay test] tier=core done in {time.perf_counter() - t0:.2f}s (exit={rc})")
        return rc

    if tier == "perf_timing":
        cmd = build_pytest_command(
            workers=1,
            marker=f"{FAST_GATE_MARKER} and perf_timing",
            durations=durations,
            keyword=keyword,
            extra_args=extra_args,
            python_exe=py,
        )
        if list_only:
            print(" ".join(cmd))
            return 0
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode

    if tier == "all":
        parallel_cmd = build_pytest_command(
            workers=workers,
            marker=f"{FAST_GATE_MARKER} and not perf_timing",
            durations=durations,
            keyword=keyword,
            extra_args=extra_args,
            python_exe=py,
        )
        serial_cmd = build_pytest_command(
            workers=1,
            marker=f"{FAST_GATE_MARKER} and perf_timing",
            durations=durations,
            keyword=keyword,
            extra_args=extra_args,
            python_exe=py,
        )
        if list_only:
            print("PARALLEL:", " ".join(parallel_cmd))
            print("SERIAL  :", " ".join(serial_cmd))
            return 0
        rc1 = subprocess.run(parallel_cmd, cwd=str(REPO_ROOT), env=env).returncode
        rc2 = subprocess.run(serial_cmd, cwd=str(REPO_ROOT), env=env).returncode
        rc = 0 if (rc1 == 0 and rc2 == 0) else 1
        print(f"[clay test] tier=all done in {time.perf_counter() - t0:.2f}s (exit={rc})")
        return rc

    # Default tier == "fast": blast-radius changed-file parallel test run
    changed = changed_python_files(base=base)
    if not changed:
        print("[clay test] no changed .py files; use `--tier core`, `--tier regression`, or `--tier all`.")
        return 0
    selected, uncovered = map_changed_to_tests(changed)
    print(f"[clay test] changed={len(changed)} mapped_tests={len(selected)} uncovered={len(uncovered)}")
    if uncovered:
        for u in uncovered[:20]:
            print(f"  !! uncovered: {u}")
    if list_only:
        for s in selected:
            print(s)
        return 0
    if not selected:
        print("[clay test] no test files mapped; run `--tier core` or add a test.")
        return 1
    base_cmd = build_pytest_command(
        workers=workers,
        marker=FAST_GATE_MARKER,
        durations=durations,
        keyword=keyword,
        python_exe=py,
    )
    rc = _exec_pytest_with_argfile(base_cmd, selected, extra_args, env=env)
    print(f"[clay test] tier=fast done in {time.perf_counter() - t0:.2f}s (exit={rc})")
    return rc

