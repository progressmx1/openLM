# Report — takeover of `offline_models` (2026-11-02)

All numbers below were measured on this machine (RTX 4070 Laptop 8 GB, Intel Arc iGPU,
Core Ultra 7 155H, Windows 11 build 26300, llama.cpp b11960). **The RX 9060 XT eGPU was not
connected during this session**, so nothing here was measured on it.

## 1. GPU placement — the "memory goes to the Intel GPU" report

**Finding: the model was on the RTX 4070 all along.** Re-running the exact launch from the log
with verbose logging: `offloaded 29/29 layers to GPU`, CUDA0 buffers 2,945 MiB weights +
777 MiB KV + 92 MiB compute; the llama-server process held **3,951 MiB dedicated on the 4070's
adapter and nothing on the Intel adapter**; the iGPU's shared usage stayed flat (1,737 →
1,709 MiB). The ~1.7 GB shown on the Intel GPU is the desktop compositor (`dwm` 666 MiB,
Explorer, the Claude/Edge apps…) because the iGPU drives the display. Full evidence: DECISIONS D1.

Real problems found and fixed:
- **No proof of placement** — llama-server's default verbosity hides the offload lines. Now
  `-lv 4` + a parser (`offload.py`) → the UI shows *“29/29 layers on NVIDIA GeForce RTX 4070
  Laptop GPU”* with weights/KV per device, and **warns** on partial or zero offload or memory on
  an unexpected device.
- **Live per-process VRAM per GPU** from the Windows GPU counters (vendor-neutral, works for
  AMD), mapped to adapters via DXGI LUIDs; nvidia-smi totals for NVIDIA (per-process is
  `[N/A]` under WDDM — verified). Shown in the status bar and on each GPU card.
- **`-ngl all` whenever a GPU is chosen**; a saved `n_gpu_layers: 0` no longer leaves the
  decision to `--fit`.
- **Index-shift trap:** without the eGPU, `Vulkan1` is the Intel iGPU (it was the 4070 with the
  eGPU). Key-only requests that now point at an integrated GPU are rejected and re-placed.
- `--no-mmap` was evaluated and **rejected with data**: `--load-mode none` raised the process's
  shared GPU memory from 120 to 566 MiB at equal speed (D2).

| Check (real hardware) | Result |
|---|---|
| f16 1.5B on “Single GPU: RTX 4070”, ctx 28,416, saved `n_gpu_layers=0` | 29/29 layers on CUDA0; process 3,951 MiB dedicated; 4070 adapter 379 → 4,330 MiB; nvidia-smi 4,325 MiB; iGPU shared 1,685 → 1,673 MiB; 37 tok/s |
| Qwen2.5-Coder-7B Q4_K_M on RTX 4070, ctx 16,384 | 29/29 layers; 5,359 MiB dedicated, 126 MiB shared; 25–31 tok/s; loads in 6–11 s |
| Backend hard-killed with a model loaded (every `agent_e2e.py` run, 14×) | llama-server gone every time (Job Object) |

## 2. GPU modes

Per model: **Auto / Single GPU / Multi-GPU / CPU**. Multi-GPU runs one Vulkan llama-server with
`--split-mode layer --tensor-split` (ratio default ∝ VRAM, slider in the UI) and reports layers
and MiB per GPU; a CUDA+AMD pick is remapped onto the Vulkan entries of the same cards. If only
one GPU is present it falls back to that GPU (fastest runtime) with a visible note. Why Vulkan
and not RPC: D3. 24 placement tests + 9 end-to-end tests against the fake server cover every rule.

| Check (real hardware) | Result |
|---|---|
| Multi-GPU requested, eGPU absent (Q8_0) | Loaded on the 4070 via CUDA with note “Multi-GPU fell back to NVIDIA GeForce RTX 4070 Laptop GPU alone (only one GPU is available)”; 29/29 layers; 59.6 tok/s |
| Split mechanism on two real GPUs (4070 + iGPU, named explicitly — the only second GPU available), ratio 3:1 | 22 layers / 1,349 MiB on the 4070 (counter: 1,363 MiB), 7 layers / 672 MiB on the iGPU (counter: 692 MiB shared); correct output; 38.8 tok/s |
| **RX 9060 XT** | **Not verified** (not connected) |

## 3. Local coding agent

Built: sessions per workspace (persistent), Ask / Plan / Agent / Auto modes, the tool set from the
brief, streamed reasoning / text / tool calls / live command output, inline approvals,
"always allow" per command prefix, destructive-command guard, step budget, interrupt, error
recovery, diff preview, per-file keep/revert, undo last turn, audit log, context compaction,
slash commands (`/help /clear /compact /model /mode /init /undo`), versioned prompt files editable
in Settings with per-model variants, environment + `AGENTS.md`/`CLAUDE.md` injection.

Making the 7B model actually act took most of the effort (D5): it never produced parseable
native tool calls, `tool_choice: required` isn't enforced for its template, but llama.cpp's
**JSON-schema constrained decoding** works — so Qwen2.5-Coder runs with constrained steps
(`thought` + tool call, or `respond`), plus loop hygiene learned from 14 real runs.

Real-model runs (Qwen2.5-Coder-7B Q4_K_M on the RTX 4070, `scripts/agent_e2e.py`, approvals given
by the harness the way a user would click; destructive commands denied):

| Run | Task | Agent version | Outcome (verified independently with `python -m pytest`) |
|---|---|---|---|
| 1–3 | wordstats CLI | native tools only | 0 tool calls: the model wrote a plan + code in Markdown and stopped |
| 4 | wordstats | + constrained escalation (bug: `response_format` was dropped) | invalid call, stopped |
| 5 | wordstats | constrained steps work | wrote files, ran pytest, looped on one edit for 30 steps; tests fail |
| 6–8, 11 | wordstats | + thought field, repeat/stuck guards, fuzzy edits, completion check | runs 6/7: 3 of 5 and 1 of 4 tests pass; run 8: test file doesn't import; run 11: 2 tests fail. The model's own tests assert wrong values (e.g. 8 words in a 6-word string) |
| 9 | **acceptance prompt** | Qwen still on native mode | ran a test file it never wrote; failed |
| 10 | **acceptance prompt** | Qwen → constrained (`variants.json`) | **pass**: `cli.py` + test written, `pytest` run by the agent (exit 0), 1 test passes |
| final-a | **acceptance prompt** | same | **fail**: model left a `try:` without `except` (→ added syntax diagnostics, D7) |
| final-b | wordstats | + cwd removal | 1/3 tests pass; stopped after completion check |
| final-c | **acceptance prompt** | + syntax diagnostics | **pass**: `cli.py` + `test_cli.py`, agent ran `unittest` (exit 0), 2 tests pass; its final small test edit was not re-run by the agent but passes |
| UI | fix planted bug in `calc.py` (demo project) | latest, through the browser UI | **pass** (details below) |

Every run: 29/29 layers on the RTX 4070, 25–33 tok/s generation, no llama-server left after the
backend was hard-killed. Raw JSON reports: `data\logserifygent-e2e-*.json`.

**Through the UI** (browser, Agent mode, demo project with a planted bug, every edit and command
approved by clicking): the agent ran the tests (1 failed) → read `calc.py` → proposed
`return a - b` → `return a + b` as a diff → re-ran the tests (exit 0) → summarised. Screenshot:
`docs/screenshots/09-edit-approval-diff.png`.

Honest limits: with this 7B model, simple tasks complete; tasks with several functions often end
with tests the model itself wrote wrong (e.g. asserting 8 words for a 6-word string) — the agent
then stops with a clear message rather than claiming success. A stronger tool-calling model is
the biggest lever.

## 4. UI

Rebuilt from scratch (React + TypeScript + Vite + Tailwind; no new runtime dependencies; system
fonts; inline SVG icons; strict CSP; everything served locally). Sessions sidebar, workspace
picker with file tree, transcript with collapsible reasoning, tool cards (command + live output,
syntax-highlighted diffs) with inline Approve/Deny/Always-allow, Plan / Changes / Files /
Terminal panels, status bar (model, GPU placement, live VRAM per GPU, tok/s, context), Models &
GPUs manager, Settings with prompt editor, Ctrl+K palette, keyboard shortcuts, dark/light,
responsive (overlay sidebar under 820 px), ARIA roles/labels. `tsc --noEmit` and `npm run build`
are clean. Playwright screenshots (installed Edge, no page errors): `docs/screenshots/01…09`.

- **Browser path verified** (`scripts\dev-backend.ps1` → http://127.0.0.1:8765, via
  `scripts\ui_test_backend.py` with a throwaway data folder).
- **Electron: not verified here.** Inside the automation's restricted job object Chromium cannot
  initialise its sandbox and `electron.exe` exits with `0x80000003` before running any app code
  (even `--version`); running it without the sandbox was (rightly) not allowed. The Electron code
  changes are small (native folder picker via IPC, title/background). Please run
  `scripts\start.ps1` once; `desktop/frontend/scripts/electron-smoke.mjs` automates the check.

## Engineering

- Backend tests: **244 passed, 1 skipped** (was 138 + 1). The suite no longer writes to `data\`
  (`backend/conftest.py` → temp `LOCAL_AI_DATA_DIR`; verified by file timestamps).
- `.gitignore` bug fixed: `models/` also hid `backend/app/models/` (all runtime code) from git (D6).
- Git: initialised, baseline commit, one commit per milestone.
- The real `data\` folder was not modified by any test or verification run (only new logs under
  `data\logs\verify\`). **On first start of the new version the app migrates `data\local_ai.db`
  to schema v4 (adds four tables, changes nothing existing).**
- Downloaded: `models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (4,683,074,336 bytes, SHA-256
  verified against Hugging Face's ETag). Dev dependency added: `playwright-core` (screenshots).

## How to run

See README “Quick start”. Short version:
`powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1` (desktop) or
`…\scripts\dev-backend.ps1` and open http://127.0.0.1:8765.

## Not done / not verified

1. RX 9060 XT: Single-GPU and Multi-GPU across 4070 + 9060 XT, hot-unplug error strings.
2. Electron launch (see above).
3. The old chat, documents (PDF/DOCX), RAG and multi-agent features are still in the backend but
   not in the new UI.
4. Agent reliability on harder tasks with the 7B model (see table).
5. Installer/packaging.

## Next steps

1. Plug in the eGPU and run `python scripts\verify_gpu.py --model models\DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf --mode multi`
   and `--mode single --gpu 9060`; paste results into DECISIONS D3.
2. Try a stronger tool-calling coder model on the 16 GB eGPU (add a `variants.json` entry; test
   `tool_format: auto` first, fall back to `json`).
3. Run `scripts\start.ps1` and click through once in Electron.
4. Bring documents/RAG into the agent as tools if still wanted.
