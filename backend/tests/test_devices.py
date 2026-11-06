"""Device discovery and placement rules (no real GPUs needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.models.devices import (
    DeviceCatalog,
    PlacementError,
    ServerBinary,
    assign_physical_keys,
    find_server_binaries,
    guess_runtime,
    is_integrated,
    parse_list_devices,
    resolve_placement,
)

VULKAN = ServerBinary("/rt/vulkan/llama-server", "vulkan")
CUDA = ServerBinary("/rt/cuda/llama-server", "cuda")

VULKAN_OUT = """ggml_vulkan: Found 3 Vulkan devices:
Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15800 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 7900 MiB free)
  Vulkan2: Intel(R) Arc(TM) Graphics (18000 MiB, 17000 MiB free)
"""
CUDA_OUT = """Available devices:
  CUDA0: NVIDIA GeForce RTX 4070 Laptop GPU (8187 MiB, 7000 MiB free)
"""

GIB = 1024 ** 3


def _devices(with_cuda: bool = True):
    devices = parse_list_devices(VULKAN_OUT, VULKAN)
    if with_cuda:
        devices += parse_list_devices(CUDA_OUT, CUDA)
    assign_physical_keys(devices)
    return devices


def _binaries(with_cuda: bool = True):
    return [VULKAN, CUDA] if with_cuda else [VULKAN]


def test_parse_list_devices() -> None:
    devices = parse_list_devices(VULKAN_OUT, VULKAN)
    assert [d.id for d in devices] == ["Vulkan0", "Vulkan1", "Vulkan2"]
    assert devices[0].name == "AMD Radeon RX 9060 XT"
    assert devices[0].total_mib == 16304 and devices[0].free_mib == 15800
    assert devices[2].name == "Intel(R) Arc(TM) Graphics"
    assert devices[2].integrated is True
    assert devices[0].runtime == "vulkan"


def test_same_card_shares_physical_key_across_runtimes() -> None:
    devices = _devices()
    nvidia = [d for d in devices if "4070" in d.name]
    assert {d.runtime for d in nvidia} == {"vulkan", "cuda"}
    assert len({d.physical_key for d in nvidia}) == 1


def test_integrated_detection() -> None:
    assert is_integrated("Intel(R) Arc(TM) Graphics")
    assert is_integrated("AMD Radeon(TM) Graphics")
    assert not is_integrated("Intel(R) Arc(TM) B580 Graphics")
    assert not is_integrated("Intel(R) Arc(TM) Pro B60 Graphics")
    assert not is_integrated("AMD Radeon RX 9060 XT")


def test_auto_prefers_biggest_free_gpu() -> None:
    placement = resolve_placement("auto", _devices(), _binaries(), 3 * GIB)
    assert placement.mode == "gpu"
    assert placement.devices[0].name == "AMD Radeon RX 9060 XT"


def test_auto_uses_other_gpu_when_first_is_busy_and_prefers_cuda_for_nvidia() -> None:
    devices = _devices()
    amd_key = next(d.physical_key for d in devices if "9060" in d.name)
    placement = resolve_placement("auto", devices, _binaries(), 3 * GIB, busy={amd_key})
    assert "4070" in placement.devices[0].name
    assert placement.runtime == "cuda"  # fastest runtime for the NVIDIA card


def test_auto_splits_when_model_is_bigger_than_any_gpu() -> None:
    placement = resolve_placement("auto", _devices(), _binaries(), 20 * GIB)
    assert placement.mode == "split"
    assert placement.runtime == "vulkan"  # only Vulkan sees both vendors
    assert [d.id for d in placement.devices] == ["Vulkan0", "Vulkan1"]  # iGPU excluded
    assert placement.tensor_split == [16304.0, 8188.0]


def test_auto_without_gpus_runs_on_cpu() -> None:
    placement = resolve_placement("auto", [], [VULKAN], 3 * GIB)
    assert placement.mode == "cpu"
    assert placement.physical_keys() == ["cpu"]


def test_explicit_device_and_cpu() -> None:
    placement = resolve_placement("cuda:CUDA0", _devices(), _binaries())
    assert placement.devices[0].key == "cuda:CUDA0"
    assert placement.binary == CUDA
    assert resolve_placement("cpu", _devices(), _binaries()).mode == "cpu"


def test_saved_placement_follows_name_when_indices_shift() -> None:
    # Saved while the 4070 was Vulkan1; now the eGPU is gone and the 4070 is Vulkan0.
    saved = {"mode": "gpu", "devices": [{"key": "vulkan:Vulkan1", "name": "NVIDIA GeForce RTX 4070 Laptop GPU"}]}
    shifted = parse_list_devices(
        "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 7900 MiB free)\n", VULKAN
    )
    assign_physical_keys(shifted)
    placement = resolve_placement(saved, shifted, [VULKAN])
    assert placement.devices[0].id == "Vulkan0"


def test_missing_saved_gpu_falls_back_to_auto_with_note() -> None:
    saved = {"mode": "gpu", "devices": [{"key": "vulkan:Vulkan0", "name": "AMD Radeon RX 9060 XT"}]}
    only_nvidia = parse_list_devices(CUDA_OUT, CUDA)
    assign_physical_keys(only_nvidia)
    placement = resolve_placement(saved, only_nvidia, [CUDA], GIB)
    assert "4070" in placement.devices[0].name
    assert placement.note and "not available" in placement.note


def test_split_across_runtimes_is_remapped_to_one_runtime() -> None:
    # The user picks "RX 9060 XT + RTX 4070 (CUDA)": the CUDA build cannot see AMD, so the
    # split runs on the Vulkan build, which sees both cards.
    placement = resolve_placement(
        {"mode": "split", "devices": ["vulkan:Vulkan0", "cuda:CUDA0"]}, _devices(), _binaries()
    )
    assert placement.mode == "split"
    assert [d.key for d in placement.devices] == ["vulkan:Vulkan0", "vulkan:Vulkan1"]


def test_split_with_no_shared_runtime_is_rejected() -> None:
    rocm = ServerBinary("/rt/rocm/llama-server", "rocm")
    amd = parse_list_devices("  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 15800 MiB free)\n", rocm)
    devices = amd + parse_list_devices(CUDA_OUT, CUDA)
    assign_physical_keys(devices)
    with pytest.raises(PlacementError, match="one runtime"):
        resolve_placement({"mode": "split", "devices": ["rocm:ROCm0", "cuda:CUDA0"]}, devices, [rocm, CUDA])


def test_no_runtime_installed_is_a_clear_error() -> None:
    with pytest.raises(PlacementError, match="install-llama-runtimes"):
        resolve_placement("auto", [], [])


def test_find_server_binaries_and_runtime_guess(tmp_path: Path) -> None:
    vulkan_dir = tmp_path / "vulkan" / "llama-b11960-bin-win-vulkan-x64"
    vulkan_dir.mkdir(parents=True)
    (vulkan_dir / "llama-server.exe").write_bytes(b"")
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir()
    (cuda_dir / "llama-server.exe").write_bytes(b"")
    found = find_server_binaries(runtimes_dir=tmp_path)
    assert sorted(b.runtime for b in found) == ["cuda", "vulkan"]
    assert guess_runtime("C:/x/llama-b1-bin-win-cuda-12.4-x64/llama-server.exe") == "cuda"


def test_catalog_reports_runner_errors() -> None:
    def runner(cmd, timeout):
        return (1, "error: failed to load vulkan-1.dll")

    catalog = DeviceCatalog(lambda: [VULKAN], runner=runner)
    assert catalog.devices(refresh=True) == []
    assert catalog.errors() and "vulkan-1.dll" in catalog.errors()[0]["error"]
