"""Placement proof parsed from real llama-server b11960 logs captured on this machine."""
from __future__ import annotations

from pathlib import Path

from app.models.offload import evaluate, layers_per_device, parse_offload_log

FIXTURES = Path(__file__).with_name("fixtures")
RTX = "NVIDIA GeForce RTX 4070 Laptop GPU"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parses_full_cuda_offload() -> None:
    report = parse_offload_log(_read("llama_cuda_f16_full.log"))
    assert report["layers_on_gpu"] == 29 and report["layers_total"] == 29
    assert report["devices"]["CUDA0"]["model_mib"] == 2944.7
    assert report["devices"]["CUDA0"]["kv_mib"] == 777.0
    assert report["devices"]["CUDA0"]["name"] == RTX
    assert report["host"]["model_mib"] == 445.1  # token embeddings stay on the host
    assert report["n_ctx"] == 28416


def test_split_log_ignores_fit_dry_run_and_reads_both_devices() -> None:
    report = parse_offload_log(_read("llama_vulkan_split_3to1.log"))
    # --fit's dry run prints 0.00 MiB buffers first; the real load must win.
    assert report["devices"]["Vulkan0"]["model_mib"] == 1043.5
    assert report["devices"]["Vulkan1"]["model_mib"] == 521.1
    assert report["devices"]["Vulkan1"]["name"] == "Intel(R) Arc(TM) Graphics"


def test_layer_assignment_matches_llama_cpp_debug_output() -> None:
    # The debug log of that run assigned layers 0-21 to Vulkan0 and 22-28 to Vulkan1.
    assert layers_per_device(29, 28, [3, 1]) == [22, 7]
    assert layers_per_device(29, 28, [1, 1]) == [15, 14]
    assert sum(layers_per_device(29, 28, [16304, 7948])) == 29
    assert layers_per_device(0, 28, [1, 1]) == [0, 0]


def test_no_offload_lines_means_unparsed() -> None:
    assert parse_offload_log("srv  load_model: loading model\n") is None


PLACE_GPU = {"mode": "gpu", "device_ids": ["CUDA0"], "device_names": [RTX], "tensor_split": None}


def test_evaluate_full_offload_is_ok() -> None:
    check = evaluate(parse_offload_log(_read("llama_cuda_f16_full.log")), PLACE_GPU)
    assert check["level"] == "ok"
    assert check["message"] == f"29/29 layers on {RTX}."
    assert check["per_device"][0]["layers"] == 29


def test_evaluate_partial_offload_warns() -> None:
    text = _read("llama_cuda_f16_full.log").replace("offloaded 29/29", "offloaded 12/29")
    check = evaluate(parse_offload_log(text), PLACE_GPU)
    assert check["level"] == "warning" and "Only 12/29" in check["message"]


def test_evaluate_zero_layers_is_an_error() -> None:
    text = _read("llama_cuda_f16_full.log").replace("offloaded 29/29", "offloaded 0/29")
    check = evaluate(parse_offload_log(text), PLACE_GPU)
    assert check["level"] == "error" and "running on the CPU" in check["message"]


def test_evaluate_memory_on_unexpected_device_is_an_error() -> None:
    report = parse_offload_log(_read("llama_vulkan_split_3to1.log"))
    only_rtx = {"mode": "gpu", "device_ids": ["Vulkan0"], "device_names": [RTX], "tensor_split": None}
    check = evaluate(report, only_rtx)
    assert check["level"] == "error" and "Vulkan1" in check["message"]


def test_evaluate_split_reports_layers_and_mib_per_gpu() -> None:
    report = parse_offload_log(_read("llama_vulkan_split_3to1.log"))
    split = {
        "mode": "split",
        "device_ids": ["Vulkan0", "Vulkan1"],
        "device_names": [RTX, "Intel(R) Arc(TM) Graphics"],
        "tensor_split": [3, 1],
    }
    check = evaluate(report, split)
    assert check["level"] == "ok"
    assert [d["layers"] for d in check["per_device"]] == [22, 7]
    assert check["per_device"][0]["model_mib"] == 1043.5
    assert "22 on" in check["message"]


def test_evaluate_without_report() -> None:
    assert evaluate(None, PLACE_GPU)["level"] == "unknown"
    assert evaluate(None, {"mode": "cpu"})["level"] == "ok"
