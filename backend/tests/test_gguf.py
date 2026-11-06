"""Tests for the dependency-free GGUF header/metadata reader."""
from __future__ import annotations

import struct
from pathlib import Path

from app.models.gguf import read_gguf_metadata, summarize_metadata


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def make_gguf(metadata: list[tuple[str, int, bytes]]) -> bytes:
    out = bytearray(struct.pack("<I", 0x46554747))  # "GGUF"
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 0)  # tensor_count
    out += struct.pack("<Q", len(metadata))  # metadata_kv_count
    for key, value_type, value in metadata:
        out += _string(key)
        out += struct.pack("<I", value_type)
        out += value
    return bytes(out)


def test_read_metadata(tmp_path: Path) -> None:
    gguf = make_gguf(
        [
            ("general.name", 8, _string("Test Model")),
            ("general.architecture", 8, _string("qwen2")),
            ("general.file_type", 4, struct.pack("<I", 7)),
            ("general.quantization_version", 4, struct.pack("<I", 2)),
            ("general.description", 8, _string("A test")),
        ]
    )
    path = tmp_path / "test.gguf"
    path.write_bytes(gguf)

    meta = read_gguf_metadata(path)
    assert meta is not None
    assert meta["general.name"] == "Test Model"
    assert meta["general.architecture"] == "qwen2"
    assert meta["general.file_type"] == 7


def test_not_a_gguf(tmp_path: Path) -> None:
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"this is not a gguf file at all")
    assert read_gguf_metadata(path) is None


def test_missing_file() -> None:
    assert read_gguf_metadata("definitely/does/not/exist.gguf") is None


def test_summarize_uses_filename_quant() -> None:
    summary = summarize_metadata(
        Path("/models/DeepSeek-R1-Q6_K_L.gguf"),
        {
            "general.name": "DeepSeek R1",
            "general.architecture": "qwen2",
            "general.file_type": 18,
        },
    )
    assert summary["quant"] == "Q6_K_L"
    assert summary["architecture"] == "qwen2"
    assert summary["name"] == "DeepSeek R1"


def test_summarize_falls_back_to_file_type() -> None:
    summary = summarize_metadata(
        Path("/models/model.gguf"),
        {"general.name": "X", "general.file_type": 1},
    )
    assert summary["quant"] == "F16"
