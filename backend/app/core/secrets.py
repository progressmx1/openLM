"""API keys for cloud model providers, encrypted at rest.

On Windows the file is encrypted with DPAPI (``CryptProtectData``): only the same Windows
user on the same machine can decrypt it, and no extra dependency is needed. Elsewhere the
file is written with owner-only permissions. Keys are never returned to the UI — only a
masked form such as ``sk-…a1b2``.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

_ENTROPY = b"local-agent-provider-keys-v1"


def _dpapi(data: bytes, encrypt: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def blob(raw: bytes) -> BLOB:
        buffer = ctypes.create_string_buffer(raw, len(raw))
        return BLOB(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))

    crypt32 = ctypes.WinDLL("crypt32")
    kernel32 = ctypes.WinDLL("kernel32")
    source, entropy, out = blob(data), blob(_ENTROPY), BLOB()
    fn = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
    # CRYPTPROTECT_UI_FORBIDDEN = 0x1
    if not fn(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0x1, ctypes.byref(out)):
        raise OSError("Windows could not " + ("encrypt" if encrypt else "decrypt") + " the provider key file.")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def mask(key: str) -> str:
    key = key.strip()
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:3]}…{key[-4:]}"


class SecretStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        raw = self.path.read_bytes()
        if sys.platform == "win32":
            raw = _dpapi(raw, encrypt=False)
        return json.loads(raw.decode("utf-8"))

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data).encode("utf-8")
        if sys.platform == "win32":
            raw = _dpapi(raw, encrypt=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        if sys.platform != "win32":
            os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            return self._read().get(name)

    def set(self, name: str, value: str) -> None:
        with self._lock:
            data = self._read()
            data[name] = value.strip()
            self._write(data)

    def delete(self, name: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(name, None) is not None:
                self._write(data)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._read())
