"""End-to-end tests of the llama-server runtime against a fake server process.

These start real child processes (a small Python stand-in for llama-server), so
they exercise process startup, health polling, per-GPU placement, streaming,
reasoning tags, crash detection, and cleanup.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from app.agents.chat_service import ChatService
from app.core.settings import SettingsStore
from app.models.adapters.base import ModelLoadError
from app.models.adapters.hybrid import HybridAdapter
from app.models.adapters.llama_server_adapter import LlamaServerAdapter
from app.models.devices import DeviceCatalog, find_server_binaries
from app.models.registry import ModelRegistry

FAKE = Path(__file__).with_name("fake_llama_server.py")


@pytest.fixture
def runtime_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "runtimes" / "vulkan"
    folder.mkdir(parents=True)
    shutil.copy(FAKE, folder / "llama_server.py")
    return tmp_path / "runtimes"


@pytest.fixture
def server_adapter(runtime_dir: Path, tmp_path: Path):
    catalog = DeviceCatalog(lambda: find_server_binaries(runtimes_dir=runtime_dir))
    adapter = LlamaServerAdapter(catalog, tmp_path / "logs", load_timeout_provider=lambda: 20)
    yield adapter
    adapter.unload_all()


@pytest.fixture
def models(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ("alpha", "beta", "gamma", "oom-model"):
        path = tmp_path / "models" / f"{name}.gguf"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"GGUF")
        paths[name] = path
    return paths


@pytest.fixture
def server_registry(repository, settings_store: SettingsStore, server_adapter, models) -> ModelRegistry:
    adapter = HybridAdapter(server_adapter, mode_provider=lambda: "auto")
    registry = ModelRegistry(repository, settings_store, adapter=adapter)
    registry.discover([str(next(iter(models.values())).parent)])
    return registry


def _id(registry: ModelRegistry, name: str) -> str:
    return next(m["id"] for m in registry.list_models() if Path(m["path"]).stem == name)


def test_devices_are_listed(server_adapter) -> None:
    devices = server_adapter.list_devices(refresh=True)
    assert [d["name"] for d in devices] == [
        "AMD Radeon RX 9060 XT",
        "NVIDIA GeForce RTX 4070 Laptop GPU",
        "Intel(R) Arc(TM) Graphics",
    ]


def test_two_models_run_on_separate_gpus(server_registry: ModelRegistry) -> None:
    a, b = _id(server_registry, "alpha"), _id(server_registry, "beta")
    status_a = server_registry.load(a)
    status_b = server_registry.load(b)

    assert status_a["state"] == "ready" and status_b["state"] == "ready"
    assert status_a["devices"] == ["vulkan:Vulkan0"]  # eGPU (biggest) first
    assert status_b["devices"] == ["vulkan:Vulkan1"]  # then the free RTX 4070
    assert status_a["n_ctx_effective"] == 8192
    # Both stay loaded: different GPUs, so no auto-unload.
    assert server_registry.adapter.is_loaded(a) and server_registry.adapter.is_loaded(b)

    reply_a = "".join(server_registry.adapter.stream_chat(a, [{"role": "user", "content": "hi"}], {}))
    reply_b = server_registry.adapter.chat(b, [{"role": "user", "content": "hi"}], {})["content"]
    assert reply_a == "<think>thinking</think>\n\ndevice=Vulkan0"
    assert reply_b == "<think>hmm</think>\n\ndevice=Vulkan1"

    devices = server_registry.devices()
    assert devices[0]["loaded_models"] == [a]
    assert devices[1]["loaded_models"] == [b]


def test_same_gpu_replaces_previous_model(server_registry: ModelRegistry) -> None:
    a, c = _id(server_registry, "alpha"), _id(server_registry, "gamma")
    server_registry.load(a, {"placement": "vulkan:Vulkan1"})
    server_registry.load(c, {"placement": "vulkan:Vulkan1"})
    assert not server_registry.adapter.is_loaded(a)
    assert server_registry.adapter.is_loaded(c)


def test_placement_is_remembered_by_name(server_registry: ModelRegistry, settings_store: SettingsStore) -> None:
    a = _id(server_registry, "alpha")
    server_registry.load(a, {"placement": "vulkan:Vulkan1"})
    profile = settings_store.load().model_profiles[a]
    assert profile["placement"]["devices"][0]["name"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
    server_registry.unload(a)
    # A plain load reuses the remembered GPU.
    assert server_registry.load(a)["devices"] == ["vulkan:Vulkan1"]


def test_split_across_both_gpus(server_registry: ModelRegistry) -> None:
    a = _id(server_registry, "alpha")
    status = server_registry.load(a, {"placement": "split"})
    assert status["devices"] == ["vulkan:Vulkan0", "vulkan:Vulkan1"]
    command = status["placement"]
    assert command["tensor_split"] == [16304.0, 8188.0]
    reply = "".join(server_registry.adapter.stream_chat(a, [{"role": "user", "content": "hi"}], {}))
    assert reply.endswith("device=Vulkan0,Vulkan1")


def test_out_of_memory_is_reported_clearly(server_registry: ModelRegistry) -> None:
    oom = _id(server_registry, "oom-model")
    with pytest.raises(ModelLoadError, match="Not enough GPU memory"):
        server_registry.load(oom)
    assert not server_registry.adapter.is_loaded(oom)


def test_crash_is_contained_and_reported(server_registry: ModelRegistry, repository, settings_store) -> None:
    a, b = _id(server_registry, "alpha"), _id(server_registry, "beta")
    server_registry.load(a)
    server_registry.load(b)

    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        list(server_registry.adapter.stream_chat(a, [{"role": "user", "content": "CRASH"}], {}))

    status = server_registry.status(a)
    assert status["state"] == "crashed" and status["loaded"] is False
    # The other GPU's model is unaffected.
    assert server_registry.adapter.chat(b, [{"role": "user", "content": "hi"}], {})["content"].endswith("Vulkan1")

    # The chat service surfaces the crash reason instead of a generic message.
    conversation = repository.create_conversation(title="t") if hasattr(repository, "create_conversation") else None
    if conversation:
        service = ChatService(repository, server_registry, settings_store)
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            service.chat(conversation["id"], a, "hello")

    # Reloading after a crash works.
    assert server_registry.load(a)["state"] == "ready"


def test_unload_stops_the_process(server_registry: ModelRegistry) -> None:
    a = _id(server_registry, "alpha")
    pid_status = server_registry.load(a)
    model = server_registry.adapter.server._models[a]
    proc = model.process
    server_registry.unload(a)
    deadline = time.monotonic() + 5
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None
    assert pid_status["pid"] == proc.pid


# ------------------------------------------------------- placement proof (section 1)
@pytest.fixture
def more_models(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ("delta-partial", "eps-noreport", "zeta-tools"):
        path = tmp_path / "models" / f"{name}.gguf"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"GGUF")
        paths[name] = path
    return paths


def _command(registry: ModelRegistry, model_id: str) -> list[str]:
    return registry.status(model_id)["command"] if "command" in registry.status(model_id) else (
        registry.adapter.server._models[model_id].config["command"]
    )


def test_chosen_gpu_gets_all_layers_even_with_saved_zero(server_registry: ModelRegistry, settings_store) -> None:
    settings = settings_store.load()
    inference = settings.inference.model_copy(update={"n_gpu_layers": 0, "n_ctx": 28416})
    settings_store.save(settings.model_copy(update={"inference": inference}))
    a = _id(server_registry, "alpha")
    status = server_registry.load(a, {"placement": {"mode": "single", "devices": [{"name": "NVIDIA GeForce RTX 4070 Laptop GPU"}]}})
    cmd = _command(server_registry, a)
    assert cmd[cmd.index("-ngl") + 1] == "all"
    assert cmd[cmd.index("-lv") + 1] == "4"
    assert "--load-mode" not in cmd  # mmap default: least shared GPU memory (D2)
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"
    check = status["placement_check"]
    assert check["level"] == "ok" and check["message"] == "29/29 layers on NVIDIA GeForce RTX 4070 Laptop GPU."
    assert status["offload"]["devices"]["Vulkan1"]["model_mib"] > 0


def test_explicit_layer_cap_is_passed_and_reported(server_registry: ModelRegistry) -> None:
    a = _id(server_registry, "alpha")
    status = server_registry.load(a, {"placement": "vulkan:Vulkan0", "n_gpu_layers": 12})
    cmd = _command(server_registry, a)
    assert cmd[cmd.index("-ngl") + 1] == "12"
    assert "--load-mode" not in cmd
    assert status["placement_check"]["level"] == "warning" and "Only 12/29" in status["placement_check"]["message"]


def test_partial_offload_shows_warning_instead_of_pretending(server_registry: ModelRegistry, more_models) -> None:
    server_registry.discover([str(next(iter(more_models.values())).parent)])
    m = _id(server_registry, "delta-partial")
    status = server_registry.load(m, {"placement": "vulkan:Vulkan1"})
    assert status["placement_check"]["level"] == "warning"
    assert "Only 10/29 layers" in status["placement_check"]["message"]


def test_missing_offload_report_is_flagged_unverified(server_registry: ModelRegistry, more_models) -> None:
    server_registry.discover([str(next(iter(more_models.values())).parent)])
    m = _id(server_registry, "eps-noreport")
    status = server_registry.load(m)
    assert status["placement_check"]["level"] == "unknown"


def test_cpu_mode_disables_gpu_offload(server_registry: ModelRegistry) -> None:
    a = _id(server_registry, "alpha")
    status = server_registry.load(a, {"placement": "cpu"})
    cmd = _command(server_registry, a)
    assert cmd[cmd.index("--device") + 1] == "none" and cmd[cmd.index("-ngl") + 1] == "0"
    assert status["placement_check"]["level"] == "ok"


def test_multi_gpu_puts_layers_on_both_gpus(server_registry: ModelRegistry) -> None:
    a = _id(server_registry, "alpha")
    status = server_registry.load(a, {"placement": {"mode": "multi", "tensor_split": [2, 1]}})
    cmd = _command(server_registry, a)
    assert cmd[cmd.index("--device") + 1] == "Vulkan0,Vulkan1"
    assert cmd[cmd.index("--tensor-split") + 1] == "2,1"
    per = status["placement_check"]["per_device"]
    assert [d["layers"] for d in per] == [20, 9]
    assert all(d["model_mib"] > 0 for d in per)
    assert status["placement_check"]["level"] == "ok"


def test_multi_gpu_falls_back_when_egpu_is_unplugged(server_registry: ModelRegistry, monkeypatch) -> None:
    monkeypatch.setenv(
        "FAKE_LLAMA_DEVICES",
        "Available devices:|  Vulkan0: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 7900 MiB free)|"
        "  Vulkan1: Intel(R) Arc(TM) Graphics (18000 MiB, 17000 MiB free)",
    )
    server_registry.adapter.server.catalog.refresh()
    a = _id(server_registry, "alpha")
    status = server_registry.load(a, {"placement": "multi"})
    assert status["state"] == "ready"
    assert status["device_names"] == ["NVIDIA GeForce RTX 4070 Laptop GPU"]
    assert "fell back" in status["placement_note"]
    assert status["placement"]["requested_mode"] == "split"


def test_stream_events_and_timings(server_registry: ModelRegistry) -> None:
    a = _id(server_registry, "alpha")
    server_registry.load(a)
    events = list(server_registry.adapter.stream_events(a, [{"role": "user", "content": "hi"}], {}))
    assert events[0] == {"type": "reasoning", "text": "thinking"}
    assert events[-1]["type"] == "done" and events[-1]["timings"]["predicted_per_second"] == 50.0
    assert server_registry.status(a)["last_timings"]["predicted_per_second"] == 50.0


def test_native_tool_call_is_streamed(server_registry: ModelRegistry, more_models) -> None:
    server_registry.discover([str(next(iter(more_models.values())).parent)])
    m = _id(server_registry, "zeta-tools")
    server_registry.load(m)
    assert server_registry.adapter.supports_native_tools(m) is True
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    events = list(
        server_registry.adapter.stream_events(
            m, [{"role": "user", "content": 'CALL read_file {"path": "a.py"}'}], {"tools": tools}
        )
    )
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls[0]["name"] == "read_file" and calls[0]["id"] == "call_1"
    assert "".join(c["arguments"] for c in calls) == '{"path": "a.py"}'
    assert events[-1]["finish_reason"] == "tool_calls"


def test_completion_payload_passes_tools_and_constrained_format() -> None:
    from app.models.adapters.llama_server_adapter import _completion_payload

    fmt = {"type": "json_schema", "json_schema": {"name": "step", "schema": {"type": "object"}}}
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    payload = _completion_payload({"response_format": fmt, "tools": tools, "max_tokens": 10})
    assert payload["response_format"] == fmt
    assert payload["tools"] == tools and payload["tool_choice"] == "auto"
    assert "response_format" not in _completion_payload({})
