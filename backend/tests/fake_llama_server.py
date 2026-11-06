"""Stand-in for llama.cpp's ``llama-server`` used by the test-suite (stdlib only).

Behaviour is driven by the model file name:
  * ``*oom*``      -> fails at startup with a Vulkan out-of-memory message
  * ``*partial*``  -> logs that only 10/29 layers were offloaded
  * ``*noreport*`` -> logs no offload lines at all
  * ``*tools*``    -> its chat template supports native tool calls
  * prompt containing ``CRASH`` -> the process exits mid-request
  * prompt containing ``CALL <tool> <json-args>`` (with ``tools`` in the request and a
    tool-capable template) -> streams a native tool call
Replies echo the ``--device`` argument so tests can check placement. Offload log lines
use the exact format of llama.cpp b11960 (see tests/fixtures/*.log).
"""
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICES = """Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15800 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 7900 MiB free)
  Vulkan2: Intel(R) Arc(TM) Graphics (18000 MiB, 17000 MiB free)
"""
NAMES = {
    "Vulkan0": "AMD Radeon RX 9060 XT",
    "Vulkan1": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "Vulkan2": "Intel(R) Arc(TM) Graphics",
}
N_LAYER = 28


def arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def log_offload(model, device):
    devices = [] if device in ("none", "auto") else device.split(",")
    for dev in devices:
        print(f"0.00.290.000 I llama_prepare_model_devices: using device {dev} ({NAMES.get(dev, dev)}) (unknown id) - 7000 MiB free")
    print(f"0.00.300.000 I llama_model_loader: - kv   5:                          qwen2.block_count u32              = {N_LAYER}")
    if "noreport" in model:
        return
    print("0.00.700.000 I load_tensors: loading model tensors, this can take a while... (load_mode = none)")
    total = N_LAYER + 1
    ngl = arg("-ngl", "auto")
    if not devices:
        on_gpu = 0
    elif "partial" in model:
        on_gpu = 10
    elif ngl in ("all", "auto"):
        on_gpu = total
    else:
        on_gpu = min(int(ngl), total)
    print(f"0.01.220.108 I load_tensors: offloaded {on_gpu}/{total} layers to GPU")
    print("0.01.220.114 I load_tensors:   CPU_Mapped model buffer size =   445.12 MiB")
    split = [float(x) for x in (arg("--tensor-split") or "").split(",") if x] or [1.0] * len(devices)
    for dev, share in zip(devices, split):
        mib = 2944.68 * share / sum(split) if on_gpu else 0.0
        print(f"0.01.220.115 I load_tensors:        {dev} model buffer size =  {mib:.2f} MiB")
    for dev, share in zip(devices, split):
        print(f"0.04.847.901 I llama_kv_cache:      {dev} KV buffer size =   {777.0 * share / sum(split):.2f} MiB")


def main():
    if "--list-devices" in sys.argv:
        # Tests simulate an unplugged eGPU by overriding the device list.
        print(os.environ.get("FAKE_LLAMA_DEVICES", DEVICES).replace("|", "\n"))
        return 0
    model = arg("-m", "")
    base = os.path.basename(model)
    port = int(arg("--port", "8080"))
    device = arg("--device", "auto")
    n_ctx = int(arg("-c", "0") or 0) or 4096
    print(f"fake llama-server: model={model} device={device}", flush=True)
    if "oom" in base:
        print("ggml_vulkan: Device memory allocation of size 9000000000 failed.", flush=True)
        print("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory", flush=True)
        return 1
    log_offload(base, device)
    sys.stdout.flush()
    started = time.monotonic()
    supports_tools = "tools" in base

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                if time.monotonic() - started < 0.3:
                    return self._json(503, {"error": {"code": 503, "message": "Loading model"}})
                return self._json(200, {"status": "ok"})
            if self.path == "/props":
                return self._json(200, {
                    "default_generation_settings": {"n_ctx": n_ctx},
                    "chat_template": "{{ tools }}" if supports_tools else "{{ messages }}",
                    "chat_template_caps": {"supports_tool_calls": supports_tools, "supports_tools": supports_tools},
                })
            self._json(404, {"error": "not found"})

        def _send_events(self, deltas, finish="stop"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in deltas:
                chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            final = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "timings": {"prompt_n": 5, "prompt_per_second": 900.0, "predicted_n": 3, "predicted_per_second": 50.0},
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            last = body["messages"][-1]
            prompt = last.get("content") or ""
            if isinstance(prompt, list):
                prompt = " ".join(part.get("text", "") for part in prompt if isinstance(part, dict))
            if "CRASH" in prompt:
                os._exit(3)
            if body.get("tools") and supports_tools and last.get("role") == "user":
                match = re.search(r"CALL (\w+) (\{.*\})", prompt)
                if match:
                    name, args = match.group(1), match.group(2)
                    half = len(args) // 2
                    deltas = [
                        {"reasoning_content": "I should use a tool."},
                        {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": args[:half]}}]},
                        {"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]},
                    ]
                    return self._send_events(deltas, finish="tool_calls")
            if last.get("role") == "tool":
                answer = f"tool said: {str(prompt)[:60]}"
            else:
                answer = f"device={device}"
            if not body.get("stream"):
                return self._json(200, {
                    "choices": [{"message": {"role": "assistant", "reasoning_content": "hmm", "content": answer}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    "timings": {"prompt_n": 3, "prompt_per_second": 800.0, "predicted_n": 2, "predicted_per_second": 40.0},
                })
            self._send_events([{"reasoning_content": "thinking"}, {"content": answer[:4]}, {"content": answer[4:]}])

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"server is listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
