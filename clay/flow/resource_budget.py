"""Serial, fail-closed process-tree memory guard for heavy physical flows."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from typing import Callable, Sequence

from clay.flow.execution_session import current_session


DEFAULT_HEAVY_MEMORY_LIMIT_BYTES = 48 * 1024**3
_HEAVY_ADMISSION_LOCK = threading.Lock()


@dataclass(frozen=True)
class ResourceBudgetResult:
    returncode: int | None
    stdout: str
    stderr: str
    peak_rss_bytes: int
    memory_limit_bytes: int
    exceeded: bool
    reason: str
    pid: int | None = None
    # The owned root process was waited for. This does not claim ownership of
    # detached descendants, which the process-table guard cannot establish.
    reaped: bool | None = None
    cleanup_errors: tuple[str, ...] = ()


def resource_budget_result_from_error(error: BaseException) -> ResourceBudgetResult | None:
    """Return actual observations retained before an original error was rethrown."""
    try:
        result = getattr(error, '_resource_budget_result', None)
    except BaseException:
        return None
    return result if type(result) is ResourceBudgetResult else None


def _retain_failure(error, process, *, peak, memory_limit_bytes, stdout_file,
                    stderr_file, terminator, grace_seconds):
    """Best-effort cleanup must neither replace the original error nor invent exit."""
    errors, stdout, stderr = [], '', ''
    reaped, returncode, pid = None, None, None
    cleanup_timeout = min(2.0, max(0.0, grace_seconds))
    def failed(stage, failure):
        note = f'{stage}: {type(failure).__name__}: {failure}'
        errors.append(note)
        try:
            error.add_note(note)
        except BaseException:
            pass  # An uncooperative exception must still be the raised object.
    if process is not None:
        reaped = False
        try:
            pid = int(process.pid)
        except BaseException as cleanup:
            failed('resource pid', cleanup)
        try:
            terminator(process, grace_seconds)
        except BaseException as cleanup:
            failed('resource termination', cleanup)
        try:
            returncode = process.wait(timeout=cleanup_timeout)
            reaped = returncode is not None
        except BaseException as cleanup:
            failed('resource wait', cleanup)
    try:
        if stdout_file is not None:
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout, stderr = stdout_file.read(), stderr_file.read()
        elif process is not None and reaped:
            stdout, stderr = process.communicate(timeout=cleanup_timeout)
    except BaseException as cleanup:
        failed('resource output capture', cleanup)
    reason = str(getattr(error, 'reason', type(error).__name__))
    result = ResourceBudgetResult(
        returncode=returncode, stdout=str(stdout or ''), stderr=str(stderr or ''),
        peak_rss_bytes=peak, memory_limit_bytes=memory_limit_bytes,
        exceeded=reason.endswith('BUDGET_EXCEEDED'), reason=reason,
        pid=pid, reaped=reaped,
        cleanup_errors=tuple(errors))
    try:
        error._resource_budget_result = result
    except BaseException as cleanup:
        failed('resource evidence attachment', cleanup)


def _linux_process_table() -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return table
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ppid = 0
        rss_kib = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
        table[int(entry.name)] = (ppid, rss_kib * 1024)
    return table


def _windows_process_table() -> dict[int, tuple[int, int]]:
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = (
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        )

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot == invalid:
        return {}
    parents: dict[int, int] = {}
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        present = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while present:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            present = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    table: dict[int, tuple[int, int]] = {}
    for pid, ppid in parents.items():
        handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            table[pid] = (ppid, 0)
            continue
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        try:
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            rss = int(counters.WorkingSetSize) if ok else 0
        finally:
            kernel32.CloseHandle(handle)
        table[pid] = (ppid, rss)
    return table


def _portable_ps_process_table() -> dict[int, tuple[int, int]]:
    try:
        completed = subprocess.run(
            ("ps", "-axo", "pid=,ppid=,rss="),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return {}
    table = {}
    for line in completed.stdout.splitlines():
        try:
            pid, ppid, rss_kib = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        table[pid] = (ppid, rss_kib * 1024)
    return table


def _process_table() -> dict[int, tuple[int, int]]:
    if os.name == "nt":
        return _windows_process_table()
    table = _linux_process_table()
    return table if table else _portable_ps_process_table()


def _descendant_pids(root_pid: int, table: dict[int, tuple[int, int]]) -> set[int]:
    selected = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _rss) in table.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def process_tree_rss_bytes(root_pid: int) -> int:
    """Sum current resident bytes for one process and every descendant."""

    table = _process_table()
    return sum(table.get(pid, (0, 0))[1] for pid in _descendant_pids(root_pid, table))


def terminate_process_tree(process: object, grace_seconds: float = 2.0) -> None:
    """Terminate observable descendants and root, then force-kill on grace expiry.

    Supported workers retain parent links and wait for their children. Detached
    or already reparented workers require an OS job/container supervisor; process
    table enumeration alone cannot retain ownership after those links disappear.
    """

    pid = int(getattr(process, "pid"))
    table = _process_table()
    descendants = sorted(_descendant_pids(pid, table) - {pid}, reverse=True)
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for child_pid in descendants:
            handle = kernel32.OpenProcess(0x0001, False, child_pid)
            if handle:
                try:
                    kernel32.TerminateProcess(handle, 1)
                finally:
                    kernel32.CloseHandle(handle)
    else:
        for child_pid in descendants:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    try:
        getattr(process, "terminate")()
    except OSError:
        pass
    try:
        getattr(process, "wait")(timeout=max(0.0, float(grace_seconds)))
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        getattr(process, "kill")()
    except OSError:
        pass
    try:
        getattr(process, "wait")(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


@contextmanager
def _admission(session):
    if session is None:
        with _HEAVY_ADMISSION_LOCK:
            yield
        return
    acquired = False
    try:
        with session.phase("resource.admission"):
            while not acquired:
                session.checkpoint()
                acquired = _HEAVY_ADMISSION_LOCK.acquire(timeout=0.05)
        session.checkpoint()
        yield
    finally:
        if acquired:
            _HEAVY_ADMISSION_LOCK.release()


def run_budgeted_command(
    command: Sequence[str],
    *,
    cwd: Path | str,
    memory_limit_bytes: int = DEFAULT_HEAVY_MEMORY_LIMIT_BYTES,
    poll_interval_s: float = 0.1,
    termination_grace_s: float = 2.0,
    process_factory: Callable[..., object] | None = None,
    rss_sampler: Callable[[int], int] = process_tree_rss_bytes,
    process_tree_terminator: Callable[[object, float], None] = terminate_process_tree,
    env: dict[str, str] | None = None,
) -> ResourceBudgetResult:
    """Run exactly one heavy command under the global serial RSS admission gate."""

    if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be a positive integer")
    if not command:
        raise ValueError("budgeted command must be non-empty")
    root = Path(cwd).resolve()
    session = current_session()
    with _admission(session), ExitStack() as files:
        stdout_file = None
        stderr_file = None
        if process_factory is None:
            stdout_file = files.enter_context(tempfile.TemporaryFile(mode="w+t", encoding="utf-8", dir=root))
            stderr_file = files.enter_context(tempfile.TemporaryFile(mode="w+t", encoding="utf-8", dir=root))
            kwargs = {
                "cwd": root,
                "text": True,
                "stdout": stdout_file,
                "stderr": stderr_file,
            }
            if env is not None:
                kwargs["env"] = dict(env)
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            factory = subprocess.Popen
        else:
            kwargs = {
                "cwd": root,
                "text": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if env is not None:
                kwargs["env"] = dict(env)
            factory = process_factory
        peak = 0
        exceeded = False
        process = None
        try:
            if session:
                session.checkpoint()
            process = factory(list(command), **kwargs)
            while True:
                if session:
                    session.checkpoint()
                rss = int(rss_sampler(int(getattr(process, "pid"))))
                if rss < 0:
                    raise ValueError(
                        "process-tree RSS sampler returned a negative value"
                    )
                peak = max(peak, rss)
                if session:
                    session.checkpoint()
                if rss > memory_limit_bytes:
                    exceeded = True
                    process_tree_terminator(process, float(termination_grace_s))
                    break
                if getattr(process, "poll")() is not None:
                    break
                if poll_interval_s > 0:
                    time.sleep(float(poll_interval_s))
            getattr(process, "wait")()
            if stdout_file is None:
                stdout, stderr = getattr(process, "communicate")()
            else:
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout, stderr = stdout_file.read(), stderr_file.read()
            returncode = int(getattr(process, "returncode", 0) or 0)
            if exceeded and returncode == 0:
                returncode = -9
            return ResourceBudgetResult(
                returncode=returncode,
                stdout=str(stdout or ""),
                stderr=str(stderr or ""),
                peak_rss_bytes=peak,
                memory_limit_bytes=memory_limit_bytes,
                exceeded=exceeded,
                reason="RESOURCE_BUDGET_EXCEEDED" if exceeded else "NONE",
                pid=int(getattr(process, "pid")),
                reaped=True,
            )
        except BaseException as error:
            try:
                _retain_failure(error, process, peak=peak, memory_limit_bytes=memory_limit_bytes,
                    stdout_file=stdout_file, stderr_file=stderr_file,
                    terminator=process_tree_terminator, grace_seconds=float(termination_grace_s))
            except BaseException as cleanup:
                try:
                    error.add_note(f'resource failure evidence unavailable: {type(cleanup).__name__}: {cleanup}')
                except BaseException:
                    pass
            raise


__all__ = [
    "DEFAULT_HEAVY_MEMORY_LIMIT_BYTES",
    "ResourceBudgetResult",
    "resource_budget_result_from_error",
    "process_tree_rss_bytes",
    "run_budgeted_command",
    "terminate_process_tree",
]
