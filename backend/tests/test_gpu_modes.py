"""GPU modes: Auto, Single GPU, Multi-GPU (one model on several GPUs), CPU.

Device lists are the real ``--list-devices`` formats from this machine, with and
without the RX 9060 XT eGPU attached (indices shift when it is unplugged).
"""
from __future__ import annotations

import pytest

from app.models.devices import (
    PlacementError,
    ServerBinary,
    assign_physical_keys,
    parse_list_devices,
    resolve_placement,
)

VULKAN = ServerBinary("/rt/vulkan/llama-server", "vulkan")
CUDA = ServerBinary("/rt/cuda/llama-server", "cuda")
GIB = 1024 ** 3

WITH_EGPU = """Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15800 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (7948 MiB, 7180 MiB free)
  Vulkan2: Intel(R) Arc(TM) Graphics (9045 MiB, 8337 MiB free)
"""
# Real output on this laptop with the eGPU unplugged: the iGPU becomes Vulkan1.
WITHOUT_EGPU = """Available devices:
  Vulkan0: NVIDIA GeForce RTX 4070 Laptop GPU (7948 MiB, 7180 MiB free)
  Vulkan1: Intel(R) Arc(TM) Graphics (9045 MiB, 8337 MiB free)
"""
CUDA_OUT = """Available devices:
  CUDA0: NVIDIA GeForce RTX 4070 Laptop GPU (8187 MiB, 7068 MiB free)
"""
RTX = "NVIDIA GeForce RTX 4070 Laptop GPU"
AMD = "AMD Radeon RX 9060 XT"
INTEL = "Intel(R) Arc(TM) Graphics"


def devices(egpu: bool = True, cuda: bool = True):
    found = parse_list_devices(WITH_EGPU if egpu else WITHOUT_EGPU, VULKAN)
    if cuda:
        found += parse_list_devices(CUDA_OUT, CUDA)
    assign_physical_keys(found)
    return found


def binaries(cuda: bool = True):
    return [VULKAN, CUDA] if cuda else [VULKAN]


# ----------------------------------------------------------------- single GPU
def test_single_gpu_by_name_uses_cuda_for_the_4070() -> None:
    p = resolve_placement({"mode": "single", "devices": [{"key": "cuda:CUDA0", "name": RTX}]}, devices(), binaries())
    assert p.mode == "gpu" and p.requested_mode == "gpu"
    assert [d.key for d in p.devices] == ["cuda:CUDA0"]
    assert p.partial_offload is False and p.note is None


def test_single_gpu_survives_index_shift_when_egpu_is_unplugged() -> None:
    # Chosen while the eGPU was attached: RTX 4070 was Vulkan1.
    request = {"mode": "gpu", "devices": [{"key": "vulkan:Vulkan1", "name": RTX}]}
    p = resolve_placement(request, devices(egpu=False, cuda=False), binaries(cuda=False))
    assert p.devices[0].name == RTX and p.devices[0].id == "Vulkan0"


def test_key_only_request_never_lands_on_the_igpu() -> None:
    # "vulkan:Vulkan1" meant the RTX 4070; without the eGPU it is the Intel iGPU.
    p = resolve_placement("vulkan:Vulkan1", devices(egpu=False, cuda=False), binaries(cuda=False), GIB)
    assert p.devices and not p.devices[0].integrated
    assert p.devices[0].name == RTX
    assert p.note and "not available" in p.note


def test_igpu_is_used_only_when_explicitly_named() -> None:
    p = resolve_placement({"mode": "gpu", "devices": [{"key": "vulkan:Vulkan2", "name": INTEL}]}, devices(), binaries())
    assert p.devices[0].name == INTEL


def test_single_without_device_picks_best_gpu_and_never_splits() -> None:
    huge = 30 * GIB
    p = resolve_placement("single", devices(), binaries(), huge)
    assert p.mode == "gpu" and len(p.devices) == 1 and p.devices[0].name == AMD
    assert p.partial_offload is True


# ------------------------------------------------------------------ multi GPU
def test_multi_gpu_splits_one_model_across_both_cards_on_vulkan() -> None:
    p = resolve_placement("multi", devices(), binaries(), 2 * GIB)
    assert p.mode == "split" and p.requested_mode == "split"
    assert p.binary == VULKAN  # CUDA cannot see the AMD card
    assert [d.name for d in p.devices] == [AMD, RTX]
    assert p.tensor_split == [16304.0, 7948.0]  # default: proportional to VRAM


def test_multi_gpu_custom_ratio() -> None:
    request = {"mode": "multi", "devices": [{"name": RTX}, {"name": AMD}], "tensor_split": [1, 3]}
    p = resolve_placement(request, devices(), binaries())
    assert [d.name for d in p.devices] == [RTX, AMD]
    assert p.tensor_split == [1.0, 3.0]
    assert {d.runtime for d in p.devices} == {"vulkan"}


def test_multi_gpu_never_includes_the_igpu_automatically() -> None:
    p = resolve_placement("multi", devices(), binaries())
    assert INTEL not in [d.name for d in p.devices]


def test_multi_gpu_falls_back_to_single_gpu_when_egpu_is_unplugged() -> None:
    p = resolve_placement("multi", devices(egpu=False), binaries(), 2 * GIB)
    assert p.mode == "gpu" and p.requested_mode == "split"
    assert [d.key for d in p.devices] == ["cuda:CUDA0"]  # fastest runtime for the 4070
    assert p.note and "fell back" in p.note and RTX in p.note


def test_multi_gpu_with_named_missing_card_falls_back_with_reason() -> None:
    request = {"mode": "split", "devices": [{"key": "vulkan:Vulkan0", "name": AMD}, {"key": "vulkan:Vulkan1", "name": RTX}]}
    p = resolve_placement(request, devices(egpu=False), binaries())
    assert p.mode == "gpu" and p.devices[0].name == RTX and p.devices[0].runtime == "cuda"
    assert AMD in p.note and "not available" in p.note


def test_multi_gpu_with_only_cuda_build_falls_back() -> None:
    p = resolve_placement("multi", parse_list_devices(CUDA_OUT, CUDA), [CUDA])
    assert p.mode == "gpu" and p.devices[0].key == "cuda:CUDA0" and "fell back" in p.note


def test_multi_gpu_with_no_gpu_falls_back_to_cpu() -> None:
    p = resolve_placement("multi", [], [VULKAN])
    assert p.mode == "cpu" and "CPU" in p.note


def test_multi_gpu_rejects_negative_ratio() -> None:
    with pytest.raises(PlacementError, match="positive"):
        resolve_placement({"mode": "multi", "devices": [{"name": AMD}, {"name": RTX}], "tensor_split": [-1, 2]}, devices(), binaries())


# ----------------------------------------------------------------- auto / cpu
def test_auto_never_uses_igpu_even_when_it_reports_more_memory() -> None:
    p = resolve_placement("auto", devices(egpu=False), binaries(), 7.5 * GIB)
    assert not any(d.integrated for d in p.devices)


def test_auto_prefers_free_gpu_then_largest() -> None:
    busy = {d.physical_key for d in devices() if d.name == AMD}
    p = resolve_placement("auto", devices(), binaries(), GIB, busy=busy)
    assert p.devices[0].name == RTX and p.devices[0].runtime == "cuda"


def test_auto_splits_when_model_exceeds_every_single_gpu() -> None:
    p = resolve_placement("auto", devices(), binaries(), 18 * GIB)
    assert p.mode == "split" and len(p.devices) == 2


def test_auto_partial_offload_only_when_nothing_fits() -> None:
    p = resolve_placement("auto", devices(), binaries(), 40 * GIB)
    assert p.mode == "gpu" and p.partial_offload is True and "CPU" in p.note


def test_cpu_mode() -> None:
    p = resolve_placement("cpu", devices(), binaries())
    assert p.mode == "cpu" and p.devices == []


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(PlacementError, match="Unknown GPU mode"):
        resolve_placement({"mode": "quantum"}, devices(), binaries())


@pytest.mark.parametrize(
    "request_, expected",
    [
        ("cpu", {"mode": "cpu"}),
        ("auto", {"mode": "auto"}),
    ],
)
def test_remembered_simple_modes(request_, expected) -> None:
    assert resolve_placement(request_, devices(), binaries()).remembered() == expected


def test_remembered_multi_keeps_names_and_ratio() -> None:
    p = resolve_placement({"mode": "multi", "tensor_split": [2, 1]}, devices(), binaries())
    saved = p.remembered()
    assert saved["mode"] == "split"
    assert [d["name"] for d in saved["devices"]] == [AMD, RTX]
    assert saved["tensor_split"] == [2.0, 1.0]
