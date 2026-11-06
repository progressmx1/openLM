# Agent handoff — Local Agent (offline coding agent + multi-GPU runtime)

Read this, then `docs/DECISIONS.md` (why things are the way they are, with measurements) and
`docs/REPORT.md` (what is verified and what is not). This replaces
`AGENT_HANDOFF_MULTI_GPU.md` (in git history).

## Machine
Windows 11 (build 26300), Acer PHN14-51, Core Ultra 7 155H, 16 GB RAM, RTX 4070 Laptop 8 GB
(Optimus; the Intel Arc iGPU drives the display), optional RX 9060 XT 16 GB eGPU over TB4.
llama.cpp b11960 in `runtimes\llama.cpp\{cuda,vulkan}` (gitignored). Python 3.10, Node 24.

## Map of the code
| Area | Files |
|---|---|
| Placement / GPU modes | `backend/app/models/devices.py` (`resolve_placement`, `_multi`, `_auto`, `_lookup`) |
| Proof of placement | `backend/app/models/offload.py` (parse `-lv 4` log, layers per device, `evaluate`) |
| Live VRAM / utilisation | `backend/app/models/gpu_telemetry.py` (DXGI LUIDs + PDH counters via ctypes, nvidia-smi totals) |
| llama-server process | `backend/app/models/adapters/llama_server_adapter.py` (`build_command`, `stream_events`, status) |
| Coding agent | `backend/app/agents/coding/` — `service.py` (loop, approvals, checkpoints, compaction, slash commands), `tools.py`, `protocol.py` (think/tool_call splitter, rescue parser), `prompts.py`, `environment.py`, `store.py` |
| Prompts | `backend/app/agents/prompts/*.md`, `variants.json` |
| HTTP | `backend/app/api/routes_coding_agent.py` (`/api/agent/*`, `/api/workspace/*`), `routes_system.py` (`/api/system/gpu-telemetry`), `routes_models.py` (`/{id}/log`) |
| UI | `desktop/frontend/src/` — `App.tsx`, `state/`, `features/{agent,panels,models,settings,sidebar,status,palette}`, `lib/transcript.ts` (event reducer) |
| DB | `backend/app/storage/db.py` migration v4 adds `agent_sessions/events/messages/checkpoints` |

## Rules that must keep holding
- Model output is never approval; only `POST /api/agent/sessions/{id}/decision` approves.
- Destructive commands always ask; compound commands are never allow-listed.
- The Intel iGPU is never chosen automatically; key-only device requests that resolve to an
  integrated GPU are rejected (index shift when the eGPU is unplugged).
- A chosen GPU gets `-ngl all`; `0`/`-1` never mean CPU.
- Tests must not touch `data\` (`backend/conftest.py` sets `LOCAL_AI_DATA_DIR` to a temp dir).
- Never modify `models\` or `data\`; use `scripts/ui_test_backend.py --data <scratch>` for
  manual UI testing.
- `.gitignore` patterns for data/models/runtimes are root-anchored (see D6) — keep them so.

## How to verify
```powershell
cd backend; python -m pytest                                  # 244 pass, 1 skip
cd desktop\frontend; npx tsc --noEmit; npm run build
python scripts\verify_gpu.py --model models\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf --mode multi
python scripts\agent_e2e.py --workspace <empty dir> --task "Create a small Python CLI with tests, run the tests, and fix failures."
```

## Open items (in priority order)
1. **Verify on the RX 9060 XT** (eGPU was not connected): `verify_gpu.py --mode multi` and
   `--mode single --gpu 9060`; check the Vulkan device order and that both adapters show VRAM.
   Check eGPU hot-unplug error strings and extend `_friendly_failure()` if they differ.
2. **Electron was not launched by the automation** (Chromium sandbox can't start inside the
   tool's job object; `electron.exe --version` exits 0x80000003). Run `scripts\start.ps1` by
   hand once; `desktop/frontend/scripts/electron-smoke.mjs` automates the check from a normal terminal.
3. Agent quality is bounded by the 7B model: simple tasks complete, multi-function tasks often
   end with failing tests (REPORT has numbers). A stronger tool-calling model (e.g. a Qwen3
   coder in the 14B class on the 16 GB eGPU) is the biggest lever; add it to `variants.json`.
4. The old chat/documents/RAG/multi-agent features still exist in the backend but the new UI
   does not expose them.
5. Packaging (installer) is still not done.
