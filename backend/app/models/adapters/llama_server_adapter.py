"""``llama-server`` adapter: one isolated server process per loaded model.

Why a process per model:

* **Stability.** A GPU driver reset, an eGPU being unplugged, or an out-of-memory
  error kills only that model's process. The backend keeps running, notices the
  exit, and reports a clear error instead of crashing the whole app.
* **Separate GPUs.** Each process is started with ``--device`` so one model can run
  on the RTX 4070 (CUDA or Vulkan) while another runs on the eGPU (Vulkan), or a
  single large model can be split across both with ``--tensor-split``.
* **Mixed vendors.** Different processes can use different builds: a CUDA build
  for the NVIDIA card and a Vulkan build for the AMD card.

The server speaks the OpenAI-compatible ``/v1/chat/completions`` API over
localhost. Reasoning returned in ``reasoning_content`` is re-wrapped in
``<think>…</think>`` so the UI's collapsible "Reasoning" section keeps working.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import httpx

from ...core import process_guard
from ...core.logging import get_logger
from ..devices import DeviceCatalog, Placement, PlacementError, resolve_placement
from ..gguf import discover_gguf_files
from ..offload import evaluate, parse_offload_log
from .base import ModelAdapter, ModelLoadError, ModelNotLoadedError
from .openai_stream import StreamState, iter_openai_events

log = get_logger(__name__)

_HEALTH_POLL_SECONDS = 0.25
_WATCHDOG_SECONDS = 2.0
_STREAM_READ_TIMEOUT = 600.0  # max silence between tokens (long prompt processing)


class _ServerModel:
    def __init__(self, model_id: str, placement: Placement, port: int, log_path: Path, config: dict[str, Any]) -> None:
        self.model_id = model_id
        self.placement = placement
        self.port = port
        self.log_path = log_path
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.log_handle: Any = None
        self.state = "loading"  # loading | ready | crashed
        self.error: Optional[str] = None
        self.n_ctx_effective: Optional[int] = None
        self.offload: Optional[dict[str, Any]] = None
        self.template_caps: dict[str, Any] = {}
        self.native_tools = False
        self.last_timings: Optional[dict[str, Any]] = None
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(connect=5.0, read=_STREAM_READ_TIMEOUT, write=30.0, pool=5.0),
            trust_env=False,  # never route localhost traffic through a proxy
        )

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except Exception:  # noqa: BLE001
                pass
            self.log_handle = None


class LlamaServerAdapter(ModelAdapter):
    runtime_name = "llama-server"

    def __init__(
        self,
        catalog: DeviceCatalog,
        log_dir: Path,
        load_timeout_provider: Optional[Any] = None,
        telemetry: Optional[Any] = None,
    ) -> None:
        self.catalog = catalog
        self.telemetry = telemetry
        self.log_dir = Path(log_dir)
        self._load_timeout = load_timeout_provider or (lambda: 600.0)
        self._models: dict[str, _ServerModel] = {}
        self._lock = threading.RLock()
        self._watchdog: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        atexit.register(self.unload_all)

    # ------------------------------------------------------------------ info
    def available(self) -> bool:
        return bool(self.catalog.binaries())

    def list_devices(self, refresh: bool = False) -> list[dict[str, Any]]:
        devices = [d.to_dict() for d in self.catalog.devices(refresh=refresh)]
        with self._lock:
            usage: dict[str, list[str]] = {}
            for model in self._models.values():
                if model.state == "crashed":
                    continue
                for key in model.placement.physical_keys():
                    usage.setdefault(key, []).append(model.model_id)
        for device in devices:
            device["loaded_models"] = usage.get(device["physical_key"], [])
        return devices

    def get_runtime_info(self) -> dict[str, Any]:
        binaries = self.catalog.binaries()
        with self._lock:
            loaded = [m for m, s in self._models.items() if s.state == "ready"]
        devices = self.catalog.cached_devices() if binaries else []
        return {
            "name": "llama-server",
            "version": None,
            "binaries": [b.to_dict() for b in binaries],
            "gpu_offload_supported": any(not d.integrated for d in devices),
            "max_devices": len(devices),
            "device_errors": self.catalog.errors(),
            "loaded_models": loaded,
        }

    def discover_models(self, locations: Iterable[str]) -> list[dict[str, Any]]:
        return discover_gguf_files(locations)

    # ------------------------------------------------------------- planning
    def _busy_keys(self, exclude: str) -> set[str]:
        with self._lock:
            return {
                key
                for model_id, model in self._models.items()
                if model_id != exclude and model.state != "crashed"
                for key in model.placement.physical_keys()
            }

    def _resolve(self, model_id: str, settings: dict[str, Any]) -> Placement:
        binaries = self.catalog.binaries()
        if not binaries:
            raise ModelLoadError(
                "No llama.cpp runtime found. Run scripts\\install-llama-runtimes.ps1 (or add a "
                "llama-server path in Settings → GPU runtimes), then try again."
            )
        # Re-scan when the cache is a few seconds old so a newly attached eGPU shows up.
        devices = self.catalog.devices(max_age=10.0)
        request = settings.get("placement")
        if request in (None, "", "auto") and settings.get("tensor_split"):
            request = {"mode": "split", "tensor_split": settings["tensor_split"]}
        try:
            return resolve_placement(
                request,
                devices,
                binaries,
                settings.get("size_bytes"),
                busy=self._busy_keys(exclude=model_id),
            )
        except PlacementError as exc:
            raise ModelLoadError(str(exc)) from exc

    def plan_load(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        placement = self._resolve(model_id, settings)
        return {**placement.to_dict(), "_placement": placement}

    # ------------------------------------------------------------- loading
    def build_command(self, placement: Placement, settings: dict[str, Any], port: int) -> list[str]:
        cmd = placement.binary.command() + [
            "-m", str(settings["path"]),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-webui",
            "-np", "1",  # one slot: predictable VRAM use, requests are serialized anyway
            # Verbosity 4 makes libllama log "offloaded N/N layers" and the per-device
            # buffer sizes, which are the proof of placement shown in the UI.
            "-lv", "4",
        ]
        if placement.devices:
            cmd += ["--device", ",".join(d.id for d in placement.devices)]
            if len(placement.devices) > 1:
                cmd += ["--split-mode", "layer"]
                if placement.tensor_split:
                    cmd += ["--tensor-split", ",".join(f"{x:g}" for x in placement.tensor_split)]
            layers = gpu_layers_arg(placement, settings)
            if layers is not None:
                cmd += ["-ngl", layers]
            # "auto" keeps llama.cpp's default (mmap). Measured on the RTX 4070 (DECISIONS D2):
            # "none" puts host-side tensors in pinned CUDA_Host memory, which Windows counts
            # as *shared GPU memory* (566 MiB vs 120 MiB with mmap) at the same speed.
            load_mode = str(settings.get("load_mode") or "auto")
            if load_mode in ("none", "mmap", "mlock"):
                cmd += ["--load-mode", load_mode]
        else:
            cmd += ["--device", "none", "-ngl", "0"]

        n_ctx = int(settings.get("n_ctx") or 0)
        if n_ctx > 0:
            cmd += ["-c", str(n_ctx)]  # 0 = let --fit choose the largest context that fits
        if settings.get("n_threads"):
            cmd += ["-t", str(int(settings["n_threads"]))]
        if settings.get("n_batch"):
            cmd += ["-b", str(int(settings["n_batch"]))]
        template = settings.get("chat_template") or settings.get("chat_format")
        if template:
            cmd += ["--chat-template", str(template)]
        return cmd

    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            existing = self._models.get(model_id)
            if existing is not None and existing.state == "ready" and existing.alive():
                return self.get_model_status(model_id)
            if existing is not None:
                self._stop(model_id)

        path = settings.get("path")
        if not path or not Path(path).exists():
            raise ModelLoadError(f"Model file not found: {path or '<no path>'}")

        plan = settings.get("placement_plan") or {}
        placement = plan.get("_placement") if isinstance(plan, dict) else None
        if not isinstance(placement, Placement):
            placement = self._resolve(model_id, settings)

        last_error: Optional[ModelLoadError] = None
        for _attempt in range(2):  # retry once on a port collision
            try:
                return self._start(model_id, placement, settings)
            except _PortInUse as exc:
                last_error = ModelLoadError(str(exc))
                continue
        raise last_error or ModelLoadError("Could not start llama-server.")

    def _start(self, model_id: str, placement: Placement, settings: dict[str, Any]) -> dict[str, Any]:
        port = _free_port()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(settings["path"]).stem[:40]
        log_path = self.log_dir / f"llama-server-{stem}.log"
        cmd = self.build_command(placement, settings, port)
        config = {
            "path": settings["path"],
            "n_ctx": int(settings.get("n_ctx") or 0) or None,
            "n_gpu_layers": settings.get("n_gpu_layers"),
            "n_threads": settings.get("n_threads"),
            "n_batch": settings.get("n_batch"),
            "command": cmd,
        }
        model = _ServerModel(model_id, placement, port, log_path, config)
        model.log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        model.log_handle.write("# " + subprocess.list2cmdline(cmd) + "\n")
        model.log_handle.flush()
        log.info("Starting llama-server for %s on %s: %s", model_id, placement.label(), cmd)
        try:
            model.process = subprocess.Popen(
                cmd,
                stdout=model.log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(Path(placement.binary.path).parent),
                env=_child_env(),
                **process_guard.popen_kwargs(),
            )
        except OSError as exc:
            model.close()
            raise ModelLoadError(f"Could not start llama-server ({placement.binary.path}): {exc}") from exc
        process_guard.guard(model.process)

        with self._lock:
            self._models[model_id] = model
        self._ensure_watchdog()

        try:
            self._wait_ready(model)
        except BaseException:
            self._stop(model_id)
            raise
        model.n_ctx_effective = self._read_context_size(model)
        model.offload = self._read_offload(model)
        model.state = "ready"
        log.info("Model %s ready on %s (port %s, ctx %s)", model_id, placement.label(), port, model.n_ctx_effective)
        return self.get_model_status(model_id)

    def _wait_ready(self, model: _ServerModel) -> None:
        deadline = time.monotonic() + float(self._load_timeout())
        while True:
            code = model.process.poll() if model.process else -1
            if code is not None:
                tail = _log_tail(model.log_path)
                if _looks_like_port_clash(tail):
                    raise _PortInUse(f"Port {model.port} was taken; retrying.")
                raise ModelLoadError(_friendly_failure(tail, model.placement, code, loading=True))
            try:
                response = model.client.get("/health", timeout=2.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise ModelLoadError(
                    f"The model did not finish loading within {int(self._load_timeout())} s on "
                    f"{model.placement.label()}. Large models over Thunderbolt can be slow to load; "
                    "raise the load timeout in Settings or use a smaller model."
                )
            time.sleep(_HEALTH_POLL_SECONDS)

    @staticmethod
    def _read_offload(model: _ServerModel) -> Optional[dict[str, Any]]:
        # The log is written by the child; give it a moment to flush after /health is ok.
        for _ in range(8):
            try:
                text = model.log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            report = parse_offload_log(text)
            if report is not None:
                return report
            time.sleep(0.25)
        return None

    @staticmethod
    def _read_context_size(model: _ServerModel) -> Optional[int]:
        try:
            props = model.client.get("/props", timeout=5.0).json()
        except Exception:  # noqa: BLE001
            return model.config.get("n_ctx")
        model.template_caps = props.get("chat_template_caps") or {}
        template = props.get("chat_template") or ""
        model.native_tools = bool(
            model.template_caps.get("supports_tool_calls")
            or model.template_caps.get("supports_tools")
            or ("tool_calls" in template or "tools" in template)
        )
        settings = props.get("default_generation_settings") or {}
        value = settings.get("n_ctx") or props.get("n_ctx")
        try:
            return int(value) if value else model.config.get("n_ctx")
        except (TypeError, ValueError):
            return model.config.get("n_ctx")

    # ------------------------------------------------------------ unloading
    def _stop(self, model_id: str) -> None:
        with self._lock:
            model = self._models.pop(model_id, None)
        if model is None:
            return
        proc = model.process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log.warning("llama-server pid %s did not exit", proc.pid)
            except OSError:
                pass
        model.close()

    def unload_model(self, model_id: str) -> None:
        self._stop(model_id)

    def unload_all(self) -> None:
        with self._lock:
            ids = list(self._models.keys())
        for model_id in ids:
            try:
                self._stop(model_id)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            model = self._models.get(model_id)
        return model is not None and model.state == "ready" and model.alive()

    # ------------------------------------------------------------ watchdog
    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(target=self._watch, name="llama-server-watchdog", daemon=True)
        self._watchdog.start()

    def _watch(self) -> None:
        while not self._shutdown.wait(_WATCHDOG_SECONDS):
            with self._lock:
                models = list(self._models.values())
            for model in models:
                self._check_crash(model)

    def _check_crash(self, model: _ServerModel) -> bool:
        if model.state != "ready" or model.process is None:
            return model.state == "crashed"
        code = model.process.poll()
        if code is None:
            return False
        model.state = "crashed"
        model.error = _friendly_failure(_log_tail(model.log_path), model.placement, code, loading=False)
        log.error("llama-server for %s exited (code %s): %s", model.model_id, code, model.error)
        model.close()
        return True

    def _require_ready(self, model_id: str) -> _ServerModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            raise ModelNotLoadedError(f"Model '{model_id}' is not loaded. Load it before generating.")
        if self._check_crash(model) or model.state == "crashed":
            raise RuntimeError(model.error or "The model process stopped. Reload the model.")
        if model.state != "ready":
            raise ModelNotLoadedError("The model is still loading.")
        return model

    def _transport_error(self, model: _ServerModel, exc: Exception) -> str:
        # Give the process a moment to exit so the crash (not the socket error) is reported.
        for _ in range(8):
            if self._check_crash(model):
                return model.error or str(exc)
            time.sleep(0.25)
        if isinstance(exc, httpx.ReadTimeout):
            return (
                f"The model on {model.placement.label()} stopped responding for "
                f"{int(_STREAM_READ_TIMEOUT)} s. Try a shorter prompt or reload the model."
            )
        return f"Lost connection to the model on {model.placement.label()}: {exc}"

    # ---------------------------------------------------------- generation
    def chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        model = self._require_ready(model_id)
        payload = {"messages": messages, "stream": False, **_completion_payload(options)}
        with model.lock:
            try:
                response = model.client.post("/v1/chat/completions", json=payload, timeout=httpx.Timeout(5.0, read=None))
            except httpx.HTTPError as exc:
                raise RuntimeError(self._transport_error(model, exc)) from exc
        if response.status_code != 200:
            raise RuntimeError(_http_error(response))
        data = response.json()
        if data.get("timings"):
            model.last_timings = _timings(data["timings"])
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        wrapper = _ThinkWrapper()
        content = wrapper.feed(message.get("reasoning_content"), message.get("content")) + wrapper.close()
        usage = data.get("usage") or {}
        return {
            "content": content,
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "timings": model.last_timings,
        }

    def stream_events(
        self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Stream structured events from ``/v1/chat/completions``.

        Yields ``{"type": "reasoning"|"content", "text"}``, ``{"type": "tool_call", "index",
        "id", "name", "arguments"}`` (argument fragments), and a final
        ``{"type": "done", "finish_reason", "usage", "timings"}``. ``options["tools"]`` is
        passed through as OpenAI tool schemas (native tool calling via ``--jinja``).
        """
        model = self._require_ready(model_id)
        payload = {"messages": messages, "stream": True, **_completion_payload(options)}
        payload["stream_options"] = {"include_usage": True}
        state = StreamState()
        with model.lock:
            try:
                with model.client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        response.read()
                        raise RuntimeError(_http_error(response))
                    yield from iter_openai_events(response.iter_lines(), state)
                    if state.timings:
                        model.last_timings = _timings(state.timings)
            except httpx.HTTPError as exc:
                raise RuntimeError(self._transport_error(model, exc)) from exc
        yield {"type": "done", "finish_reason": state.finish_reason, "usage": state.usage, "timings": model.last_timings}

    def stream_chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[str]:
        wrapper = _ThinkWrapper()
        for event in self.stream_events(model_id, messages, options):
            if event["type"] == "reasoning":
                text = wrapper.feed(event["text"], None)
            elif event["type"] == "content":
                text = wrapper.feed(None, event["text"])
            else:
                continue
            if text:
                yield text
        tail = wrapper.close()
        if tail:
            yield tail

    def supports_native_tools(self, model_id: str) -> bool:
        with self._lock:
            model = self._models.get(model_id)
        return bool(model and model.native_tools)

    def process_ids(self) -> dict[int, str]:
        with self._lock:
            return {
                m.process.pid: m.model_id
                for m in self._models.values()
                if m.process is not None and m.state != "crashed"
            }

    def _vram(self, model: _ServerModel) -> Optional[list[dict[str, Any]]]:
        if self.telemetry is None or model.process is None:
            return None
        try:
            snap = self.telemetry.snapshot(self.process_ids())
        except Exception:  # noqa: BLE001
            return None
        if not snap.get("supported"):
            return None
        usage = []
        for adapter in snap.get("adapters", []):
            for proc in adapter.get("processes", []):
                if proc["pid"] == model.process.pid:
                    usage.append(
                        {
                            "adapter": adapter["name"],
                            "luid": adapter["luid"],
                            "integrated": adapter["integrated"],
                            "dedicated_mib": proc["dedicated_mib"],
                            "shared_mib": proc["shared_mib"],
                            "utilization_pct": proc.get("utilization_pct"),
                        }
                    )
        return usage

    # --------------------------------------------------------------- status
    def get_model_status(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            return {"loaded": False}
        self._check_crash(model)
        placement = model.placement.to_dict()
        return {
            "loaded": model.state == "ready",
            "state": model.state,
            "error": model.error,
            "runtime": "llama-server",
            "backend": placement["runtime"],
            "placement": placement,
            "devices": placement["devices"],
            "device_names": placement["device_names"],
            "physical_devices": placement["physical_devices"],
            "placement_note": placement["note"],
            "n_ctx_effective": model.n_ctx_effective,
            "offload": model.offload,
            "placement_check": evaluate(model.offload, placement, log_ready=model.state != "loading"),
            "vram": self._vram(model),
            "last_timings": model.last_timings,
            "native_tools": model.native_tools,
            "port": model.port,
            "pid": model.process.pid if model.process else None,
            "log_path": str(model.log_path),
            **model.config,
        }


# ---------------------------------------------------------------- helpers
class _PortInUse(Exception):
    pass


class _ThinkWrapper:
    """Re-wrap ``reasoning_content`` deltas in ``<think>`` tags for the UI."""

    def __init__(self) -> None:
        self.in_think = False

    def feed(self, reasoning: Optional[str], content: Optional[str]) -> str:
        out: list[str] = []
        if reasoning:
            if not self.in_think:
                out.append("<think>")
                self.in_think = True
            out.append(reasoning)
        if content:
            if self.in_think:
                out.append("</think>\n\n")
                self.in_think = False
            out.append(content)
        return "".join(out)

    def close(self) -> str:
        if self.in_think:
            self.in_think = False
            return "</think>"
        return ""


def gpu_layers_arg(placement: Placement, settings: dict[str, Any]) -> Optional[str]:
    """``-ngl`` value. A chosen GPU gets every layer; ``0``/``-1`` never mean "CPU".

    * user cap ``n_gpu_layers > 0`` -> that number (reported as a partial offload)
    * auto placement of a model known to be larger than VRAM -> ``None`` (llama.cpp's
      ``--fit`` decides how many layers fit)
    * otherwise -> ``"all"``
    """
    try:
        cap = int(settings.get("n_gpu_layers") or 0)
    except (TypeError, ValueError):
        cap = 0
    if cap > 0:
        return str(cap)
    if placement.partial_offload:
        return None
    return "all"


def _timings(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_n": raw.get("prompt_n"),
        "prompt_per_second": raw.get("prompt_per_second"),
        "predicted_n": raw.get("predicted_n"),
        "predicted_per_second": raw.get("predicted_per_second"),
        "cache_n": raw.get("cache_n"),
    }


def _completion_payload(options: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature": float(options.get("temperature", 0.7)),
        "top_p": float(options.get("top_p", 0.95)),
        "top_k": int(options.get("top_k", 40)),
        "repeat_penalty": float(options.get("repeat_penalty", 1.1)),
        "max_tokens": int(options.get("max_tokens", 2048)),
        "cache_prompt": True,
    }
    seed = options.get("seed", -1)
    if seed is not None and int(seed) >= 0:
        payload["seed"] = int(seed)
    stop = options.get("stop")
    if stop:
        payload["stop"] = stop
    if options.get("tools"):
        payload["tools"] = options["tools"]
        payload["tool_choice"] = options.get("tool_choice", "auto")
        payload["parallel_tool_calls"] = bool(options.get("parallel_tool_calls", False))
    if options.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = options["chat_template_kwargs"]
    if options.get("response_format"):
        # JSON-schema constrained decoding (llama.cpp compiles it to a grammar).
        payload["response_format"] = options["response_format"]
    return payload


def _http_error(response: httpx.Response) -> str:
    try:
        body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        message = err.get("message") if isinstance(err, dict) else (err or body)
    except Exception:  # noqa: BLE001
        message = response.text[:500]
    text = str(message)
    if "context" in text.lower() and ("exceed" in text.lower() or "size" in text.lower()):
        return f"The conversation is longer than the model's context window. Start a new chat or raise the context length. ({text})"
    return f"Model server error ({response.status_code}): {text}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # The server only talks to localhost; keep proxies from interfering.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    return env


def _log_tail(path: Path, lines: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "\n".join(handle.read().splitlines()[-lines:])
    except OSError:
        return ""


def _looks_like_port_clash(tail: str) -> bool:
    text = tail.lower()
    return "address already in use" in text or "couldn't bind" in text or "failed to bind" in text


def _friendly_failure(tail: str, placement: Placement, code: Optional[int], loading: bool) -> str:
    text = tail.lower()
    where = placement.label()
    if any(s in text for s in ("out of memory", "outofdevicememory", "failed to allocate", "cudamalloc failed", "unable to allocate")):
        return (
            f"Not enough GPU memory on {where}. Try a smaller quantization, a lower context length, "
            "another GPU, or splitting the model across GPUs."
        )
    if any(s in text for s in ("devicelost", "device lost", "device_lost", "device removed", "tdr")):
        return (
            f"The GPU ({where}) stopped responding. If it is the eGPU, check the Thunderbolt cable and "
            "power, reconnect it, click Refresh devices, and load the model again."
        )
    if any(s in text for s in ("invalid device", "unknown device", "no device", "device not found")):
        return f"The selected GPU ({where}) was not found. The eGPU may be disconnected — click Refresh devices."
    if any(s in text for s in ("failed to load model", "invalid magic", "gguf_init", "unknown model architecture")):
        last = tail.strip().splitlines()[-1] if tail.strip() else ""
        return f"llama.cpp could not load this model file. It may be corrupt or need a newer runtime. ({last})"
    prefix = "The model failed to start" if loading else "The model process stopped unexpectedly"
    # Skip the command line we write at the top of each log.
    lines = [line.strip() for line in tail.strip().splitlines() if line.strip() and not line.startswith("# ")]
    details = " | ".join(lines[-4:])
    hint = "" if loading else " If this model was on the eGPU, check that it is still connected, then reload."
    message = f"{prefix} on {where} (exit code {code}).{hint}"
    return f"{message} Details: {details}" if details else message
