"""Minimal, dependency-free GGUF header/metadata reader.

Only the header and its metadata key/value pairs are parsed, so discovery never
loads model weights into memory. Parsing is defensive: any malformed or
unexpected value is skipped and the function still returns what it found.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# Map ``general.file_type`` codes to human-readable quantization labels.
_FILETYPE_LABELS = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    5: "Q8_0",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q2_K",
    9: "Q3_K_S",
    10: "Q3_K_M",
    11: "Q3_K_L",
    12: "Q4_K_S",
    13: "Q4_K_M",
    14: "Q5_K_S",
    15: "Q5_K_M",
    16: "Q6_K",
    17: "IQ2_XXS",
    18: "IQ2_XS",
    19: "IQ3_XXS",
    20: "IQ1_S",
    21: "IQ4_NL",
    22: "IQ3_S",
    23: "IQ3_M",
    24: "IQ2_S",
    25: "IQ2_M",
    26: "IQ4_XS",
    27: "IQ1_M",
    28: "BF16",
    29: "Q4_0_4_4",
    30: "Q4_0_4_8",
    31: "Q4_0_8_8",
    32: "TQ1_0",
    33: "TQ2_0",
    34: "IQ4_NL_4_4",
    35: "IQ4_NL_4_8",
    36: "IQ4_NL_8_8",
    37: "Q4_K_L",
    38: "Q6_K_L",
}

_FIXED_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
    6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}
_STRUCT = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected end of file")
    return data


def _read_string(f) -> str:
    (length,) = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _read_value(f, value_type: int) -> Any:
    if value_type in (0, 1, 2, 3, 4, 5, 10, 11):
        size = _FIXED_SIZES[value_type]
        fmt = _STRUCT[size]
        (value,) = struct.unpack(fmt, _read_exact(f, size))
        return value
    if value_type == 6:  # float32
        return struct.unpack("<f", _read_exact(f, 4))[0]
    if value_type == 12:  # float64
        return struct.unpack("<d", _read_exact(f, 8))[0]
    if value_type == 7:  # bool
        return struct.unpack("<?", _read_exact(f, 1))[0]
    if value_type == 8:  # string
        return _read_string(f)
    if value_type == 9:  # array
        (elem_type,) = struct.unpack("<I", _read_exact(f, 4))
        (length,) = struct.unpack("<Q", _read_exact(f, 8))
        return [_read_value(f, elem_type) for _ in range(length)]
    raise ValueError(f"unknown value type {value_type}")


def read_gguf_metadata(path: str | Path) -> Optional[dict[str, Any]]:
    """Read GGUF metadata keys of interest. Returns None if not a valid GGUF."""
    path = Path(path)
    metadata: dict[str, Any] = {}
    version: Optional[int] = None
    try:
        with path.open("rb") as f:
            (magic,) = struct.unpack("<I", _read_exact(f, 4))
            if magic != GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", _read_exact(f, 4))
            (_tensor_count,) = struct.unpack("<Q", _read_exact(f, 8))
            (kv_count,) = struct.unpack("<Q", _read_exact(f, 8))
            for _ in range(kv_count):
                key = _read_string(f)
                (value_type,) = struct.unpack("<I", _read_exact(f, 4))
                value = _read_value(f, value_type)
                metadata[key] = value
    except (OSError, EOFError, struct.error, ValueError):
        # Defensive: return what was collected so far (possibly empty).
        if not metadata:
            return None
    metadata["_gguf_version"] = version
    return metadata


def summarize_metadata(path: str | Path, metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return discovery metadata used for registration."""
    path = Path(path)
    name = metadata.get("general.name") if metadata else None
    architecture = metadata.get("general.architecture") if metadata else None
    file_type = metadata.get("general.file_type") if metadata else None
    quant_label = _FILETYPE_LABELS.get(file_type) if isinstance(file_type, int) else None
    # Filename-derived quant label is more specific for *_K_L variants.
    quant_label = _quant_from_filename(path.name) or quant_label or "unknown"
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = None
    return {
        "name": str(name) if name else path.stem,
        "architecture": architecture if isinstance(architecture, str) else None,
        "quant": quant_label,
        "file_type": file_type if isinstance(file_type, int) else None,
        "size_bytes": size_bytes,
        "chat_template": metadata.get("tokenizer.chat_template") if metadata else None,
        "description": metadata.get("general.description") if metadata else None,
    }


def _quant_from_filename(filename: str) -> Optional[str]:
    import re

    match = re.search(r"-(Q\d[_\w]*|F16|BF16|IQ\d[\w_]*)(?:_L)?(?=\.|$)", filename, re.I)
    if not match:
        return None
    return match.group(1).upper().replace("_K_L", "_K_L")


def discover_gguf_files(locations) -> list[dict[str, Any]]:
    """Scan folders (recursively) for ``.gguf`` files and summarize each one.

    Multi-part shards other than the first (``-00002-of-00003.gguf``) are skipped,
    since llama.cpp loads the remaining parts from the first one.
    """
    import re

    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for location in locations:
        base = Path(location)
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob("*.gguf")
        for path in paths:
            if path.suffix.lower() != ".gguf":
                continue
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", path.name, re.IGNORECASE)
            if shard and shard.group(1) != "00001":
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                metadata = read_gguf_metadata(path)
            except Exception:  # noqa: BLE001 - one bad file must not abort the scan
                metadata = None
            summary = summarize_metadata(path, metadata)
            if shard:
                # Report the size of the whole model, not just the first part.
                prefix = path.name[: shard.start()]
                parts = path.parent.glob(f"{prefix}-*-of-*.gguf")
                summary["size_bytes"] = sum(p.stat().st_size for p in parts if p.is_file())
            summary["path"] = resolved
            summary["format"] = "gguf"
            discovered.append(summary)
    return discovered
