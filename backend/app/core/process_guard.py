"""Make child processes die with the backend.

If the backend is killed abruptly (Electron's ``kill()`` on Windows is a hard
``TerminateProcess``), any ``llama-server`` children would otherwise keep running
and keep holding GPU memory. On Windows every child is placed in a Job Object
flagged ``KILL_ON_JOB_CLOSE``; when the backend dies, Windows closes the job handle
and terminates the children. On Linux, ``PR_SET_PDEATHSIG`` does the same.

Everything here is best-effort: failure to set up the guard never blocks a launch.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from typing import Any, Callable, Optional

from .logging import get_logger

log = get_logger(__name__)

_job_handle: Optional[int] = None
_job_lock = threading.Lock()


def _create_windows_job() -> Optional[int]:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        handle, 9, ctypes.byref(info), ctypes.sizeof(info)  # 9 = JobObjectExtendedLimitInformation
    )
    if not ok:
        kernel32.CloseHandle(handle)
        return None
    return handle


def _assign_windows(proc: subprocess.Popen) -> None:
    global _job_handle
    from ctypes import wintypes

    with _job_lock:
        if _job_handle is None:
            _job_handle = _create_windows_job()
        if not _job_handle:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        handle = int(getattr(proc, "_handle", 0) or 0)
        if handle and not kernel32.AssignProcessToJobObject(_job_handle, handle):
            log.warning("Could not attach pid %s to the job object (error %s)", proc.pid, ctypes.get_last_error())


def popen_kwargs() -> dict[str, Any]:
    """Extra ``Popen`` arguments for a guarded, window-less child process."""
    if sys.platform == "win32":
        # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        return {"creationflags": 0x08000000 | 0x00000200}
    if sys.platform.startswith("linux"):
        return {"preexec_fn": _linux_pdeathsig()}
    return {}


def _linux_pdeathsig() -> Callable[[], None]:
    def _set() -> None:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM
        except Exception:  # noqa: BLE001
            pass

    return _set


def guard(proc: subprocess.Popen) -> None:
    """Tie ``proc``'s lifetime to this process (best-effort)."""
    if sys.platform != "win32":
        return
    try:
        _assign_windows(proc)
    except Exception as exc:  # noqa: BLE001
        log.warning("Process guard unavailable: %s", exc)
