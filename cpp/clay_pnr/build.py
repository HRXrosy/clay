#!/usr/bin/env python3
"""Parallel, content-hash incremental C++20 builder for Clay PnR native libraries.

Architecture:
    1. `clay_pnr_core` (pure C++20 static core objects — compiled WITHOUT Python
       include paths so any accidental <Python.h> leak fails at compile time!)
    2. Thin CPython extension modules installed atomically into `clay/_native/`:
       - `_place_cpp`      (src/place/py_place_module.cpp + core placement objs)
       - `_pathfinder_cpp` (src/route/py_pathfinder_module.cpp + core pathfinder objs)
       - `_gridless_cpp`   (src/route/gridless_{spatial,search}_py.cpp + py_gridless_module.cpp + core gridless objs)
    3. Pure C++20 native unit test binary (`clay_pnr_cpp_tests[.exe]`):
       - Compiled and linked with ZERO Python include/library dependencies.

Usage:
    python cpp/clay_pnr/build.py --target all --test
    python cpp/clay_pnr/build.py --target place
    python cpp/clay_pnr/build.py --target tests --test
    python cpp/clay_pnr/build.py --force --werror --test
    python -m clay build --target all --test
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
INCLUDE_DIR = ROOT / "include"
BUILD_ROOT = ROOT / "build" / "modular"
OUT_DIR = REPO_ROOT / "clay" / "_native"
MANIFEST_PATH = BUILD_ROOT / "build_manifest.json"


# Pure C++20 core translation units (compiled WITHOUT Python include paths).
CORE_SOURCES: dict[str, tuple[str, ...]] = {
    "core": (
        "src/core/polygon_set.cpp",
        "src/core/symbol_table.cpp",
        "src/gds/gds_stream.cpp",
        "src/layout/layout_db.cpp",
        "src/pcell/mos_pcell.cpp",
        "src/ir/clay_parser.cpp",
        "src/engine/standalone_synth.cpp",
    ),
    "place": (
        "src/place/sweep_union.cpp",
        "src/place/bvh_index.cpp",
        "src/place/seqpair.cpp",
        "src/place/compactor.cpp",
        "src/place/envelope_placer.cpp",
        "src/place/banded_placer.cpp",
    ),
    "pathfinder": (
        "src/route/pathfinder_core.cpp",
    ),
    "gridless": (
        "src/route/gridless_core.cpp",
        "src/route/gridless_spatial.cpp",
        "src/route/gridless_search.cpp",
    ),
}

# Thin CPython extension modules (linked against their corresponding core objs).
EXT_SPECS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "engine": (
        "_engine_cpp",
        ("src/engine/py_engine_module.cpp",),
        (
            *CORE_SOURCES["core"],
            *CORE_SOURCES["place"],
            "src/route/gridless_spatial.cpp",
        ),
    ),
    "place": (
        "_place_cpp",
        ("src/place/py_place_module.cpp",),
        CORE_SOURCES["place"],
    ),
    "pathfinder": (
        "_pathfinder_cpp",
        ("src/route/py_pathfinder_module.cpp",),
        CORE_SOURCES["pathfinder"],
    ),
    "gridless": (
        "_gridless_cpp",
        (
            "src/route/gridless_spatial_py.cpp",
            "src/route/gridless_search_py.cpp",
            "src/route/py_gridless_module.cpp",
        ),
        CORE_SOURCES["gridless"],
    ),
}

# Pure C++20 native unit tests (zero Python dependency).
TEST_SOURCES: tuple[str, ...] = (
    "tests/test_main.cpp",
    "tests/test_core_geometry.cpp",
    "tests/test_core_polygon_gds.cpp",
    "tests/test_place_kernels.cpp",
    "tests/test_pathfinder_core.cpp",
    "tests/test_gridless_core.cpp",
    "tests/test_engine_phase23.cpp",
    "tests/test_ir_and_synth.cpp",
)

# Standalone pure C++20 CLI binary (zero Python dependency).
CLI_SOURCES: tuple[str, ...] = (
    "src/cli/clay_engine_main.cpp",
)

ALL_CORE_SOURCES: tuple[str, ...] = (
    *CORE_SOURCES["core"],
    *CORE_SOURCES["place"],
    *CORE_SOURCES["pathfinder"],
    *CORE_SOURCES["gridless"],
)


@dataclass(frozen=True)
class CompileJob:
    rel_src: str
    src_path: Path
    obj_path: Path
    needs_python: bool
    key: str


def _find_vcvars64() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _capture_msvc_env() -> dict[str, str]:
    """Resolve MSVC vcvars64.bat environment once per build invocation."""
    vcvars = _find_vcvars64()
    if vcvars is None:
        raise RuntimeError("MSVC vcvars64.bat not found (Visual Studio 2022 x64 required)")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    probe_bat = BUILD_ROOT / "_probe_vcvars64.bat"
    probe_bat.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        "set\r\n",
        encoding="utf-8",
    )
    probe = subprocess.run(
        ["cmd.exe", "/d", "/c", str(probe_bat)],
        capture_output=True,
        text=True,
        check=True,
    )
    env: dict[str, str] = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k:
                env[k] = v
    return env


def _python_lib_path() -> Path:
    base = Path(sys.base_prefix)
    py_ver = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    for candidate in (base / "libs", Path(sys.prefix) / "libs"):
        if candidate.is_dir():
            exact = candidate / py_ver
            if exact.is_file():
                return exact
            libs = sorted(candidate.glob("python3*.lib"))
            if libs:
                return libs[0]
    raise FileNotFoundError(f"Could not find {py_ver} under {base} or {sys.prefix}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _headers_digest() -> str:
    """Compute a single deterministic digest over all headers under include/ and tests/*.hpp."""
    h = hashlib.sha256()
    header_paths = sorted(INCLUDE_DIR.rglob("*.hpp")) + sorted((ROOT / "tests").glob("*.hpp"))
    for hp in header_paths:
        rel = hp.relative_to(ROOT).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hp.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _obj_path_for(rel_src: str, is_windows: bool) -> Path:
    ext = ".obj" if is_windows else ".o"
    slug = rel_src.replace("/", "__").replace("\\", "__").removesuffix(".cpp")
    return BUILD_ROOT / "obj" / f"{slug}{ext}"


def _tu_fingerprint(
    src_path: Path,
    headers_hash: str,
    flags_sig: str,
    needs_python: bool,
) -> str:
    h = hashlib.sha256()
    h.update(_sha256_file(src_path).encode("ascii"))
    h.update(b"|")
    h.update(headers_hash.encode("ascii"))
    h.update(b"|")
    h.update(flags_sig.encode("utf-8"))
    h.update(b"|py=" + (b"1" if needs_python else b"0"))
    return h.hexdigest()


def _compile_tu_windows(
    job: CompileJob,
    msvc_env: dict[str, str],
    py_inc: Path,
    werror: bool,
) -> tuple[str, str]:
    job.obj_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cl.exe",
        "/nologo",
        "/c",
        "/std:c++20",
        "/O2",
        "/Oi",
        "/Ot",
        "/arch:AVX2",
        "/EHsc",
        "/MD",
        "/DNDEBUG",
        "/W4",
        "/permissive-",
        f"/I{INCLUDE_DIR}",
    ]
    if werror:
        cmd.append("/WX")
    if job.needs_python:
        cmd.append(f"/I{py_inc}")
    cmd.extend([str(job.src_path), f"/Fo{job.obj_path}"])
    res = subprocess.run(cmd, cwd=str(BUILD_ROOT), env=msvc_env, capture_output=True, text=True)
    combined = "\n".join(s for s in (res.stdout.strip(), res.stderr.strip()) if s)
    if res.returncode != 0:
        raise RuntimeError(f"MSVC compile failed for {job.rel_src} (exit {res.returncode}):\n{combined}")
    return job.rel_src, combined


def _compile_tu_posix(
    job: CompileJob,
    cxx: str,
    py_inc: Path,
    werror: bool,
) -> tuple[str, str]:
    job.obj_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx,
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        f"-I{INCLUDE_DIR}",
    ]
    if werror:
        cmd.append("-Werror")
    if job.needs_python:
        cmd.append(f"-I{py_inc}")
    cmd.extend(["-c", str(job.src_path), "-o", str(job.obj_path)])
    res = subprocess.run(cmd, cwd=str(BUILD_ROOT), capture_output=True, text=True)
    combined = "\n".join(s for s in (res.stdout.strip(), res.stderr.strip()) if s)
    if res.returncode != 0:
        raise RuntimeError(f"{cxx} compile failed for {job.rel_src} (exit {res.returncode}):\n{combined}")
    return job.rel_src, combined


def _link_ext_windows(
    mod_name: str,
    objs: list[Path],
    msvc_env: dict[str, str],
    py_lib: Path,
) -> Path:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    link_dir = BUILD_ROOT / "link" / mod_name
    link_dir.mkdir(parents=True, exist_ok=True)
    tmp_pyd = link_dir / f"{mod_name}{ext_suffix}"
    out_pyd = OUT_DIR / f"{mod_name}{ext_suffix}"

    cmd = [
        "link.exe",
        "/NOLOGO",
        "/DLL",
        f"/OUT:{tmp_pyd}",
        f"/IMPLIB:{link_dir / (mod_name + '.lib')}",
        *[str(o) for o in objs],
        str(py_lib),
    ]
    res = subprocess.run(cmd, cwd=str(link_dir), env=msvc_env, capture_output=True, text=True)
    combined = "\n".join(s for s in (res.stdout.strip(), res.stderr.strip()) if s)
    if res.returncode != 0:
        raise RuntimeError(f"MSVC link failed for {mod_name} (exit {res.returncode}):\n{combined}")
    if combined:
        print(combined, flush=True)
    shutil.copy2(tmp_pyd, out_pyd)
    return out_pyd


def _link_ext_posix(mod_name: str, objs: list[Path], cxx: str) -> Path:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_so = OUT_DIR / f"{mod_name}{ext_suffix}"
    cmd = [cxx, "-shared", *[str(o) for o in objs], "-o", str(out_so)]
    res = subprocess.run(cmd, cwd=str(BUILD_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Link failed for {mod_name}:\n{res.stdout}\n{res.stderr}")
    return out_so


def _link_exe_windows(exe_name: str, objs: list[Path], msvc_env: dict[str, str]) -> Path:
    bin_dir = BUILD_ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    out_exe = bin_dir / f"{exe_name}.exe"
    cmd = [
        "link.exe",
        "/NOLOGO",
        f"/OUT:{out_exe}",
        *[str(o) for o in objs],
    ]
    res = subprocess.run(cmd, cwd=str(bin_dir), env=msvc_env, capture_output=True, text=True)
    combined = "\n".join(s for s in (res.stdout.strip(), res.stderr.strip()) if s)
    if res.returncode != 0:
        raise RuntimeError(f"MSVC link failed for {exe_name} (exit {res.returncode}):\n{combined}")
    if combined:
        print(combined, flush=True)
    return out_exe


def _link_exe_posix(exe_name: str, objs: list[Path], cxx: str) -> Path:
    bin_dir = BUILD_ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    out_exe = bin_dir / exe_name
    cmd = [cxx, *[str(o) for o in objs], "-o", str(out_exe)]
    res = subprocess.run(cmd, cwd=str(bin_dir), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Link failed for {exe_name}:\n{res.stdout}\n{res.stderr}")
    return out_exe


def test_binary_path() -> Path:
    name = "clay_pnr_cpp_tests.exe" if platform.system() == "Windows" else "clay_pnr_cpp_tests"
    return BUILD_ROOT / "bin" / name


def cli_binary_path() -> Path:
    name = "clay_engine_cli.exe" if platform.system() == "Windows" else "clay_engine_cli"
    return BUILD_ROOT / "bin" / name


def build_targets(
    *,
    target: str = "all",
    jobs: int | None = None,
    force: bool = False,
    clean: bool = False,
    werror: bool = False,
    run_tests: bool = False,
    test_filter: str | None = None,
) -> int:
    t0 = time.perf_counter()
    is_windows = platform.system() == "Windows"

    if clean and BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
        print(f"[clay_pnr] cleaned {BUILD_ROOT}", flush=True)

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {} if force else _load_manifest()
    headers_hash = _headers_digest()
    flags_sig = f"win={is_windows}|c++20|W4|permissive-|werror={werror}|py={sys.version_info.major}.{sys.version_info.minor}"

    selected_exts: list[str] = []
    build_core_only = False
    build_tests = False
    build_cli = False
    if target == "all":
        selected_exts = ["engine", "place", "pathfinder", "gridless"]
        build_tests = True
        build_cli = True
    elif target == "core":
        build_core_only = True
    elif target == "tests":
        build_tests = True
    elif target == "cli":
        build_cli = True
    else:
        selected_exts = [target]

    if run_tests:
        build_tests = True

    # Collect unique TUs to compile.
    tu_map: dict[str, bool] = {}  # rel_src -> needs_python
    if build_core_only:
        for src in ALL_CORE_SOURCES:
            tu_map[src] = False
    for ext_key in selected_exts:
        _, py_srcs, core_srcs = EXT_SPECS[ext_key]
        for src in core_srcs:
            tu_map[src] = False
        for src in py_srcs:
            tu_map[src] = True
    if build_tests:
        for src in ALL_CORE_SOURCES:
            tu_map[src] = False
        for src in TEST_SOURCES:
            tu_map[src] = False
    if build_cli:
        for src in ALL_CORE_SOURCES:
            tu_map[src] = False
        for src in CLI_SOURCES:
            tu_map[src] = False

    pending_jobs: list[CompileJob] = []
    tu_fingerprints: dict[str, str] = {}
    for rel_src, needs_py in sorted(tu_map.items()):
        src_path = ROOT / rel_src
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing C++20 source file: {src_path}")
        obj_path = _obj_path_for(rel_src, is_windows)
        fp = _tu_fingerprint(src_path, headers_hash, flags_sig, needs_py)
        tu_fingerprints[rel_src] = fp
        if force or not obj_path.is_file() or manifest.get(f"obj:{rel_src}") != fp:
            pending_jobs.append(CompileJob(rel_src, src_path, obj_path, needs_py, fp))

    msvc_env: dict[str, str] | None = None
    cxx = ""
    py_inc = Path(sysconfig.get_path("include"))
    py_lib: Path | None = None

    def ensure_toolchain() -> None:
        nonlocal msvc_env, cxx, py_lib
        if is_windows:
            if msvc_env is None:
                msvc_env = _capture_msvc_env()
            if selected_exts and py_lib is None:
                py_lib = _python_lib_path()
        else:
            if not cxx:
                cxx = shutil.which("g++") or shutil.which("clang++") or "c++"

    recompiled_srcs: set[str] = set()
    if pending_jobs:
        ensure_toolchain()
        max_workers = max(1, jobs or min(len(pending_jobs), os.cpu_count() or 4))
        print(
            f"[clay_pnr] compiling {len(pending_jobs)}/{len(tu_map)} C++20 TU(s) "
            f"(jobs={max_workers}, W4/permissive-{' + WX' if werror else ''})",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            if is_windows:
                assert msvc_env is not None
                futs = [
                    pool.submit(_compile_tu_windows, j, msvc_env, py_inc, werror)
                    for j in pending_jobs
                ]
            else:
                futs = [
                    pool.submit(_compile_tu_posix, j, cxx, py_inc, werror)
                    for j in pending_jobs
                ]
            for fut in as_completed(futs):
                rel_src, output = fut.result()
                recompiled_srcs.add(rel_src)
                manifest[f"obj:{rel_src}"] = tu_fingerprints[rel_src]
                if output:
                    print(output, flush=True)
    else:
        print(f"[clay_pnr] all {len(tu_map)} C++20 TU(s) up to date (content-hash cache hit)", flush=True)

    # Link selected Python extension modules.
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if is_windows else ".so")
    for ext_key in selected_exts:
        mod_name, py_srcs, core_srcs = EXT_SPECS[ext_key]
        needed_srcs = (*core_srcs, *py_srcs)
        out_bin = OUT_DIR / f"{mod_name}{ext_suffix}"
        link_sig = hashlib.sha256(
            "|".join(f"{s}:{tu_fingerprints[s]}" for s in needed_srcs).encode("ascii")
        ).hexdigest()
        needs_link = (
            force
            or not out_bin.is_file()
            or manifest.get(f"ext:{mod_name}") != link_sig
            or any(s in recompiled_srcs for s in needed_srcs)
        )
        if not needs_link:
            print(f"[clay_pnr] UP-TO-DATE {out_bin.name} ({out_bin.stat().st_size} bytes)", flush=True)
            continue
        ensure_toolchain()
        objs = [_obj_path_for(s, is_windows) for s in needed_srcs]
        if is_windows:
            assert msvc_env is not None and py_lib is not None
            out_path = _link_ext_windows(mod_name, objs, msvc_env, py_lib)
        else:
            out_path = _link_ext_posix(mod_name, objs, cxx)
        manifest[f"ext:{mod_name}"] = link_sig
        print(f"[clay_pnr] OK {out_path} ({out_path.stat().st_size} bytes)", flush=True)

    # Link pure C++20 native unit test binary.
    test_exe = test_binary_path()
    if build_tests:
        test_needed = (*ALL_CORE_SOURCES, *TEST_SOURCES)
        test_sig = hashlib.sha256(
            "|".join(f"{s}:{tu_fingerprints[s]}" for s in test_needed).encode("ascii")
        ).hexdigest()
        needs_test_link = (
            force
            or not test_exe.is_file()
            or manifest.get("bin:clay_pnr_cpp_tests") != test_sig
            or any(s in recompiled_srcs for s in test_needed)
        )
        if needs_test_link:
            ensure_toolchain()
            objs = [_obj_path_for(s, is_windows) for s in test_needed]
            if is_windows:
                assert msvc_env is not None
                test_exe = _link_exe_windows("clay_pnr_cpp_tests", objs, msvc_env)
            else:
                test_exe = _link_exe_posix("clay_pnr_cpp_tests", objs, cxx)
            manifest["bin:clay_pnr_cpp_tests"] = test_sig
            print(f"[clay_pnr] OK {test_exe} ({test_exe.stat().st_size} bytes)", flush=True)
        else:
            print(f"[clay_pnr] UP-TO-DATE {test_exe.name}", flush=True)

    # Link standalone pure C++20 CLI binary (zero Python dependency).
    cli_exe = cli_binary_path()
    if build_cli:
        cli_needed = (*ALL_CORE_SOURCES, *CLI_SOURCES)
        cli_sig = hashlib.sha256(
            "|".join(f"{s}:{tu_fingerprints[s]}" for s in cli_needed).encode("ascii")
        ).hexdigest()
        needs_cli_link = (
            force
            or not cli_exe.is_file()
            or manifest.get("bin:clay_engine_cli") != cli_sig
            or any(s in recompiled_srcs for s in cli_needed)
        )
        if needs_cli_link:
            ensure_toolchain()
            objs = [_obj_path_for(s, is_windows) for s in cli_needed]
            if is_windows:
                assert msvc_env is not None
                cli_exe = _link_exe_windows("clay_engine_cli", objs, msvc_env)
            else:
                cli_exe = _link_exe_posix("clay_engine_cli", objs, cxx)
            manifest["bin:clay_engine_cli"] = cli_sig
            print(f"[clay_pnr] OK {cli_exe} ({cli_exe.stat().st_size} bytes)", flush=True)
        else:
            print(f"[clay_pnr] UP-TO-DATE {cli_exe.name}", flush=True)

    _save_manifest(manifest)
    print(f"[clay_pnr] build finished in {time.perf_counter() - t0:.2f}s", flush=True)

    if run_tests:
        cmd = [str(test_exe)]
        if test_filter:
            cmd.append(test_filter)
        print(f"[clay_pnr] running pure C++20 unit tests: {test_exe.name}", flush=True)
        t_test = time.perf_counter()
        res = subprocess.run(cmd, cwd=str(ROOT))
        print(
            f"[clay_pnr] C++20 native tests exited {res.returncode} in {(time.perf_counter() - t_test) * 1000:.1f} ms",
            flush=True,
        )
        return res.returncode

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Clay C++20 PnR modular libraries & native tests.")
    parser.add_argument(
        "--target",
        choices=["all", "core", "engine", "place", "pathfinder", "gridless", "tests", "cli"],
        default="all",
        help="Target(s) to build (default: all).",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Parallel compiler jobs (default: auto).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore content-hash freshness cache and recompile all selected TUs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove cpp/clay_pnr/build/modular before building.",
    )
    parser.add_argument(
        "--werror",
        action="store_true",
        help="Treat compiler warnings as errors (/WX or -Werror).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Build and execute pure C++20 native unit tests after compiling.",
    )
    parser.add_argument(
        "--filter",
        dest="test_filter",
        default=None,
        help="Optional substring filter for C++20 native unit tests.",
    )
    args = parser.parse_args(argv)
    return build_targets(
        target=args.target,
        jobs=args.jobs,
        force=args.force,
        clean=args.clean,
        werror=args.werror,
        run_tests=args.test,
        test_filter=args.test_filter,
    )


if __name__ == "__main__":
    raise SystemExit(main())
