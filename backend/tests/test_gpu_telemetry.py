"""Telemetry aggregation from Windows GPU counters (pure function; values from this machine)."""
from __future__ import annotations

import sys

import pytest

from app.models.gpu_telemetry import Adapter, GpuTelemetry, build_snapshot, query_nvidia_smi

MIB = 1024 * 1024
RTX_LUID = "0x00000000_0x00015117"
IGPU_LUID = "0x00000000_0x00014c79"

ADAPTERS = [
    Adapter(RTX_LUID, "NVIDIA GeForce RTX 4070 Laptop GPU", 0x10DE, 7948),
    Adapter(IGPU_LUID, "Intel(R) Arc(TM) Graphics", 0x8086, 128),
    Adapter("0x00000000_0x00015089", "Microsoft Basic Render Driver", 0x1414, 0, software=True),
]

# Measured with the f16 model on CUDA0 (llama-server pid 29460).
COUNTERS = {
    "proc_dedicated": {
        f"pid_29460_luid_{RTX_LUID}_phys_0": 3951 * MIB,
        f"pid_2220_luid_{IGPU_LUID}_phys_0": 0,
    },
    "proc_shared": {
        f"pid_29460_luid_{RTX_LUID}_phys_0": 120 * MIB,
        f"pid_2220_luid_{IGPU_LUID}_phys_0": 666 * MIB,
    },
    "adapter_dedicated": {f"luid_{RTX_LUID}_phys_0": 4356 * MIB, f"luid_{IGPU_LUID}_phys_0": 0},
    "adapter_shared": {f"luid_{RTX_LUID}_phys_0": 173 * MIB, f"luid_{IGPU_LUID}_phys_0": 1709 * MIB},
    "engine_util": {
        f"pid_29460_luid_{RTX_LUID}_phys_0_eng_1_engtype_Compute_0": 87.5,
        f"pid_2220_luid_{IGPU_LUID}_phys_0_eng_0_engtype_3D": 12.0,
        f"pid_99_luid_{IGPU_LUID}_phys_0_eng_0_engtype_3D": 95.0,
    },
}


def test_snapshot_attributes_model_memory_to_the_right_adapter() -> None:
    snap = build_snapshot(ADAPTERS, COUNTERS, {29460: "model-a"})
    by_name = {a["name"]: a for a in snap["adapters"]}
    assert "Microsoft Basic Render Driver" not in by_name  # software adapter hidden
    rtx = by_name["NVIDIA GeForce RTX 4070 Laptop GPU"]
    assert rtx["dedicated_used_mib"] == 4356
    assert rtx["processes"] == [
        {"pid": 29460, "model_id": "model-a", "dedicated_mib": 3951, "shared_mib": 120, "utilization_pct": 87.5}
    ]
    assert rtx["utilization"]["compute_0"] == 87.5
    igpu = by_name["Intel(R) Arc(TM) Graphics"]
    assert igpu["integrated"] is True
    assert igpu["processes"] == []  # dwm is not one of our models
    assert igpu["shared_used_mib"] == 1709
    assert igpu["utilization"]["3d"] == 100.0  # summed over processes, capped


def test_nvidia_smi_totals_are_attached_by_name() -> None:
    nvidia = query_nvidia_smi(lambda cmd: "0, NVIDIA GeForce RTX 4070 Laptop GPU, 4325, 8188, 97\n")
    snap = build_snapshot(ADAPTERS, COUNTERS, {}, nvidia)
    rtx = next(a for a in snap["adapters"] if a["vendor"] == "nvidia")
    assert rtx["nvidia_smi"] == {
        "index": 0, "name": "NVIDIA GeForce RTX 4070 Laptop GPU", "used_mib": 4325, "total_mib": 8188, "utilization_pct": 97
    }


def test_nvidia_smi_missing_is_empty() -> None:
    def boom(cmd):
        raise FileNotFoundError("nvidia-smi")

    assert query_nvidia_smi(boom) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows GPU counters")
def test_live_snapshot_on_windows_does_not_raise() -> None:
    snap = GpuTelemetry().snapshot({})
    assert "adapters" in snap
    telemetry = GpuTelemetry()
    telemetry.close()


def test_unsupported_platform(monkeypatch) -> None:
    telemetry = GpuTelemetry()
    monkeypatch.setattr(type(telemetry), "supported", property(lambda self: False))
    assert telemetry.snapshot()["supported"] is False
