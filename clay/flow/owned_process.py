"""Windows Job-owned command execution; no Docker daemon cleanup is implied."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable


@dataclass(frozen=True)
class OwnedProcessResult:
    process_id: int
    exit_code: int
    timed_out: bool
    descendants_terminated: bool
    job_active_after: int
    daemon_cleanup: str = "UNKNOWN"


def run_owned_process(argv, *, stdout_path, stderr_path, timeout_seconds,
                      checkpoint: Callable[[], None] | None = None,
                      cwd=None, cleanup_seconds=5.0, natural_exit_grace_seconds=0.0) -> OwnedProcessResult:
    """Run one command in a kill-on-close Job, capturing unmodified output bytes.

    The deadline covers the command and its descendants. A checkpoint exception
    is propagated only after the Job is terminated and observed empty. Output
    paths must be new, distinct files. Docker daemon containers are not members
    of the Job; their cleanup requires a separate ownership-aware procedure.
    """
    if os.name != "nt":
        raise NotImplementedError("owned process execution requires Windows Job Objects")
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(type(a) is not str or not a or "\x00" in a for a in argv)):
        raise ValueError("argv must contain nonempty strings without NUL")
    from clay.flow.execution_budget import _validate_seconds
    timeout_seconds = _validate_seconds(timeout_seconds, "timeout_seconds")
    cleanup_seconds = _validate_seconds(cleanup_seconds, "cleanup_seconds")
    if cleanup_seconds > 60:
        raise ValueError("cleanup_seconds must not exceed 60 seconds")
    if (type(natural_exit_grace_seconds) not in (int, float)
            or not math.isfinite(natural_exit_grace_seconds)
            or not 0 <= natural_exit_grace_seconds <= .5):
        raise ValueError('natural exit observation must be between 0 and 0.5 seconds')
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable")
    out, err = Path(stdout_path).absolute(), Path(stderr_path).absolute()
    if os.path.normcase(str(out)) == os.path.normcase(str(err)):
        raise ValueError("stdout and stderr paths must be distinct")
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(argv[0])
    if checkpoint:
        checkpoint()
    return _run_windows(tuple(argv), executable, out, err, timeout_seconds,
                        cleanup_seconds, checkpoint, cwd, natural_exit_grace_seconds)


def _run_windows(argv, executable, out, err, timeout_seconds, cleanup_seconds, checkpoint, cwd,
                 natural_exit_grace_seconds):
    import ctypes
    import msvcrt
    from contextlib import contextmanager
    from ctypes import wintypes

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
    class BASIC(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                    ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', wintypes.DWORD),
                    ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]
    class EXTENDED(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
    class ACCOUNTING(ctypes.Structure):
        _fields_ = [('TotalUserTime', ctypes.c_longlong), ('TotalKernelTime', ctypes.c_longlong),
                    ('ThisPeriodTotalUserTime', ctypes.c_longlong), ('ThisPeriodTotalKernelTime', ctypes.c_longlong),
                    ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                    ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]
    class STARTUPINFO(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('lpReserved', wintypes.LPWSTR), ('lpDesktop', wintypes.LPWSTR),
                    ('lpTitle', wintypes.LPWSTR), ('dwX', wintypes.DWORD), ('dwY', wintypes.DWORD),
                    ('dwXSize', wintypes.DWORD), ('dwYSize', wintypes.DWORD), ('dwXCountChars', wintypes.DWORD),
                    ('dwYCountChars', wintypes.DWORD), ('dwFillAttribute', wintypes.DWORD),
                    ('dwFlags', wintypes.DWORD), ('wShowWindow', wintypes.WORD), ('cbReserved2', wintypes.WORD),
                    ('lpReserved2', ctypes.POINTER(ctypes.c_byte)), ('hStdInput', wintypes.HANDLE),
                    ('hStdOutput', wintypes.HANDLE), ('hStdError', wintypes.HANDLE)]
    class STARTUPINFOEX(ctypes.Structure):
        _fields_ = [('startup', STARTUPINFO), ('attributes', ctypes.c_void_p)]
    class PROCESS(ctypes.Structure):
        _fields_ = [('hProcess', wintypes.HANDLE), ('hThread', wintypes.HANDLE),
                    ('dwProcessId', wintypes.DWORD), ('dwThreadId', wintypes.DWORD)]

    k = ctypes.WinDLL('kernel32', use_last_error=True)
    def bind(name, args, result):
        fn = getattr(k, name); fn.argtypes = args; fn.restype = result
        return fn
    handle, dword, boolean, ptr = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, ctypes.c_void_p
    create_job = bind('CreateJobObjectW', (ptr, wintypes.LPCWSTR), handle)
    set_job = bind('SetInformationJobObject', (handle, ctypes.c_int, ptr, dword), boolean)
    assign = bind('AssignProcessToJobObject', (handle, handle), boolean)
    create = bind('CreateProcessW', (wintypes.LPCWSTR, wintypes.LPWSTR, ptr, ptr, boolean, dword, ptr,
                                   wintypes.LPCWSTR, ptr, ctypes.POINTER(PROCESS)), boolean)
    resume = bind('ResumeThread', (handle,), dword)
    wait = bind('WaitForSingleObject', (handle, dword), dword)
    exit_code = bind('GetExitCodeProcess', (handle, ctypes.POINTER(dword)), boolean)
    terminate = bind('TerminateJobObject', (handle, wintypes.UINT), boolean)
    terminate_process = bind('TerminateProcess', (handle, wintypes.UINT), boolean)
    query = bind('QueryInformationJobObject', (handle, ctypes.c_int, ptr, dword, ptr), boolean)
    close = bind('CloseHandle', (handle,), boolean)
    init_attrs = bind('InitializeProcThreadAttributeList', (ptr, dword, dword, ctypes.POINTER(ctypes.c_size_t)), boolean)
    update_attrs = bind('UpdateProcThreadAttribute', (ptr, dword, ctypes.c_size_t, ptr, ctypes.c_size_t, ptr, ptr), boolean)
    delete_attrs = bind('DeleteProcThreadAttributeList', (ptr,), None)
    def checked(ok, label):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error(), label)
        return ok
    def active(job):
        info = ACCOUNTING()
        checked(query(job, 1, ctypes.byref(info), ctypes.sizeof(info), None), 'QueryInformationJobObject')
        return info.ActiveProcesses
    def drain(job):
        until = time.monotonic() + cleanup_seconds
        while active(job):
            if time.monotonic() >= until:
                raise RuntimeError('JOB_CLEANUP_UNVERIFIED: active processes remain')
            time.sleep(min(.05, max(0.0, until-time.monotonic())))

    job = checked(create_job(None, None), 'CreateJobObjectW')
    process = PROCESS()
    assigned = False
    attr_ptr = None
    attrs_initialized = False
    handle_cleanup_failures = []
    deadline = time.monotonic() + timeout_seconds
    @contextmanager
    def managed_file(opener):
        from clay.flow.control_stop import record_unwind_failure
        stream = opener()
        primary = None
        try:
            yield stream
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                stream.close()
            except BaseException as ending:
                failure = RuntimeError('OWNED_PROCESS_CLEANUP_UNVERIFIED: close process stream')
                failure.__cause__ = ending
                handle_cleanup_failures.append(failure)
                if primary is not None:
                    record_unwind_failure(primary, failure)
                else:
                    raise failure from ending
    try:
        limits = EXTENDED(); limits.BasicLimitInformation.LimitFlags = 0x2000
        checked(set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)), 'SetInformationJobObject')
        with managed_file(lambda: open(os.devnull, 'rb')) as stdin, \
             managed_file(lambda: out.open('xb')) as stdout, \
             managed_file(lambda: err.open('xb')) as stderr:
            handles = [msvcrt.get_osfhandle(f.fileno()) for f in (stdin, stdout, stderr)]
            enabled_handles = []
            handle_primary = None
            try:
                for h in handles:
                    os.set_handle_inheritable(h, True)
                    enabled_handles.append(h)
                size = ctypes.c_size_t()
                init_attrs(None, 1, 0, ctypes.byref(size))
                if not size.value:
                    raise ctypes.WinError(ctypes.get_last_error(), 'InitializeProcThreadAttributeList')
                attr_buffer = ctypes.create_string_buffer(size.value)
                attr_ptr = ctypes.cast(attr_buffer, ptr)
                checked(init_attrs(attr_ptr, 1, 0, ctypes.byref(size)), 'InitializeProcThreadAttributeList')
                attrs_initialized = True
                inherited = (handle * 3)(*handles)
                checked(update_attrs(attr_ptr, 0, 0x20002, ctypes.cast(inherited, ptr),
                                     ctypes.sizeof(inherited), None, None), 'UpdateProcThreadAttribute')
                si = STARTUPINFOEX(); si.startup.cb = ctypes.sizeof(si); si.startup.dwFlags = 0x100
                si.startup.hStdInput, si.startup.hStdOutput, si.startup.hStdError = handles
                si.attributes = attr_ptr
                command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
                checked(create(executable, command, None, None, True, 0x4 | 0x08000000 | 0x00080000,
                               None, str(Path(cwd).resolve()) if cwd is not None else None,
                               ctypes.byref(si), ctypes.byref(process)), 'CreateProcessW')
            except BaseException as error:
                handle_primary = error
                raise
            finally:
                from clay.flow.control_stop import record_unwind_failure
                restore_primary = None
                for h in enabled_handles:
                    try:
                        os.set_handle_inheritable(h, False)
                    except BaseException as ending:
                        failure = RuntimeError('OWNED_PROCESS_CLEANUP_UNVERIFIED: restore inheritable handle')
                        failure.__cause__ = ending
                        handle_cleanup_failures.append(failure)
                        if handle_primary is not None:
                            record_unwind_failure(handle_primary, failure)
                        elif restore_primary is None:
                            restore_primary = failure
                        else:
                            record_unwind_failure(restore_primary, failure)
                if handle_primary is None and restore_primary is not None:
                    raise restore_primary
            if not assign(job, process.hProcess):
                raise ctypes.WinError(ctypes.get_last_error(), 'AssignProcessToJobObject')
            assigned = True
            if checkpoint:
                checkpoint()
            if time.monotonic() >= deadline:
                timed_out = True
                checked(terminate(job, 124), 'TerminateJobObject')
            else:
                if resume(process.hThread) == 0xffffffff:
                    raise ctypes.WinError(ctypes.get_last_error(), 'ResumeThread')
                timed_out = False
                while True:
                    if checkpoint:
                        checkpoint()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        checked(terminate(job, 124), 'TerminateJobObject')
                        break
                    wait_ms = 50 if remaining >= .05 else max(1, math.ceil(remaining*1000))
                    status = wait(process.hProcess, wait_ms)
                    if status == 0:
                        break
                    if status != 0x102:
                        raise RuntimeError('WaitForSingleObject failed: ' + hex(status))
            if wait(process.hProcess, max(1, math.ceil(cleanup_seconds*1000))) != 0:
                raise RuntimeError('process termination was not observed')
            code = dword(); checked(exit_code(process.hProcess, ctypes.byref(code)), 'GetExitCodeProcess')
            remaining = active(job)
            grace_end = min(deadline, time.monotonic() + natural_exit_grace_seconds)
            while remaining and time.monotonic() < grace_end:
                if checkpoint:
                    checkpoint()
                time.sleep(min(.05, max(0.0, grace_end-time.monotonic())))
                remaining = active(job)
            descendants_terminated = bool(remaining)
            if remaining:
                checked(terminate(job, 125), 'TerminateJobObject descendants')
            drain(job)
            if checkpoint:
                checkpoint()
            if time.monotonic() >= deadline:
                timed_out = True
            return OwnedProcessResult(process.dwProcessId, code.value, timed_out, descendants_terminated, 0)
    except BaseException as error:
        from clay.flow.control_stop import record_unwind_failure

        failures = list(handle_cleanup_failures)
        def attempt(label, action):
            try:
                return action()
            except BaseException as ending:
                failure = RuntimeError('OWNED_PROCESS_CLEANUP_UNVERIFIED: ' + label)
                failure.__cause__ = ending
                failures.append(failure)
                record_unwind_failure(error, failure)
                return None
        if assigned:
            def stop_job():
                checked(terminate(job, 125), 'TerminateJobObject on error')
                return True
            if attempt('terminate Job', stop_job) is None:
                attempt('retry terminate Job', stop_job)
            attempt('verify Job empty', lambda: drain(job) or True)
        elif process.hProcess:
            # The process is suspended until assignment; never resume it.
            def stop_process():
                if wait(process.hProcess, 0) != 0:
                    checked(terminate_process(process.hProcess, 1), 'TerminateProcess on error')
                return True
            if attempt('terminate unassigned process', stop_process) is None:
                attempt('retry terminate unassigned process', stop_process)
            def verify_process():
                if wait(process.hProcess, max(1, math.ceil(cleanup_seconds*1000))) != 0:
                    raise RuntimeError('unassigned suspended process cleanup unverified')
                return True
            attempt('verify unassigned process exit', verify_process)
        if process.hProcess:
            def get_code():
                code = dword()
                checked(exit_code(process.hProcess, ctypes.byref(code)), 'GetExitCodeProcess on error')
                return code.value
            code = attempt('read process exit code', get_code)
            count = attempt('read Job active count', lambda: active(job))
            try:
                error.owned_process = {
                    'process_id': process.dwProcessId, 'exit_code': code,
                    'assigned': assigned, 'job_active_after': count,
                    'cleanup_status': 'UNVERIFIED' if failures else 'CLEAN',
                    'stdout_path': str(out), 'stderr_path': str(err),
                    'daemon_cleanup': 'UNKNOWN',
                }
            except BaseException:
                pass
        raise
    finally:
        if attrs_initialized:
            delete_attrs(attr_ptr)
        if process.hThread:
            close(process.hThread)
        if process.hProcess:
            close(process.hProcess)
        close(job)
