You are taking over my project `C:\Users\progr\Documents\personal_projects\offline_models` and you have full control of it. I am away, so work autonomously: make the decisions yourself, keep going until everything below is done and verified, and only stop for something destructive or irreversible. Record every significant decision in `docs/DECISIONS.md` as you go.

## 0. Before you change anything
1. Read `docs/AGENT_HANDOFF_MULTI_GPU.md` (architecture + current state), `README.md`, and skim `local_ai_desktop_agent_prompt.md` (original spec).
2. This folder is not a git repo. Run `git init`, check that `.gitignore` excludes `models/`, `data/`, `runtimes/`, `node_modules/`, and commit the current state as a baseline. Commit after each milestone so anything can be rolled back.
3. Never delete or modify anything in `models/` or `data/` (except logs you create). PowerShell scripts here must be run with `powershell -ExecutionPolicy Bypass -File <script>` (unsigned scripts are blocked).

## Machine
- Windows 11, Acer Predator Helios Neo 14, Intel Core Ultra 7 155H, 16 GB RAM (soldered).
- Internal NVIDIA RTX 4070 Laptop GPU (8 GB) — Optimus laptop, the Intel Arc iGPU drives the display.
- eGPU over Thunderbolt 4: AMD Radeon RX 9060 XT (16 GB). It may or may not be connected while you work; support both.
- llama.cpp runtimes already installed (build b11960): `runtimes\llama.cpp\cuda\` (sees only the 4070) and `runtimes\llama.cpp\vulkan\` (sees AMD + NVIDIA + Intel iGPU).
- Models in `models\`: DeepSeek-R1-Distill-Qwen-1.5B (f16, Q8_0, Q6_K_L).

You are running on this machine, so verify on the real hardware (nvidia-smi, Task Manager counters, the real llama-server.exe), not just with mocks.

## 1. TOP PRIORITY — fix and PROVE GPU placement
**Symptom:** I select the RTX 4070 for a model, but Task Manager shows the memory going to the Intel GPU.

What I know from the logs: the app launched `runtimes\llama.cpp\cuda\llama-server.exe -m ...DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf --device CUDA0 -c 28416 -b 512` (no `-ngl`, relying on llama.cpp's automatic `--fit`). Generation ran at ~52 tok/s and prompt eval at ~358 tok/s. The server log contains no line confirming how many layers were offloaded or which buffers went to CUDA, so nothing currently proves where the model ran. Saved settings have `n_gpu_layers: 0`, `n_ctx: 28416`, `max_tokens: 12288`.

Do this:
1. **Find the truth first.** Load a model on the 4070 and measure with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and the Windows counters `Get-Counter '\GPU Process Memory(*)\Dedicated Usage'` / `'\GPU Engine(*engtype_Compute*)\Utilization Percentage'` (map LUIDs to adapters). Run llama-server with higher log verbosity to capture the offload/buffer lines. Determine whether the cause is (a) layers not actually offloaded (e.g. `--fit` choices, the ngl=0 setting, the large context), (b) model weights mmapped in shared/system memory, (c) the wrong device/build being used, (d) Task Manager showing the iGPU because it drives the display/Electron while CUDA compute is hidden under the "Cuda/Compute" engine, or something else. Write the findings into `docs/DECISIONS.md`.
2. **Fix it.** When the user picks a GPU, all layers must go to that GPU (pass explicit `-ngl` = all unless the user set a cap; never let a saved `0` mean CPU silently). Consider `--no-mmap` when fully offloaded so weights are not counted in shared memory. The Intel iGPU must never be used unless the user explicitly chooses it.
3. **Prove it in the app.** After load, parse the llama-server log (or `/props`, `/metrics`) to report "N/N layers on <GPU>", and show live per-process VRAM per GPU (nvidia-smi for NVIDIA; the Windows GPU Process Memory counters for AMD/any vendor). If the actual placement differs from what the user chose (e.g. 0 layers offloaded), show a clear warning instead of pretending.
4. Acceptance: with a model "on RTX 4070", nvidia-smi lists the llama-server process using VRAM roughly the model size + KV cache, the UI shows it, and the iGPU's dedicated/shared usage does not grow by the model size.

## 2. GPU modes: single GPU or one model across two GPUs
- Add a clear per-model choice: **Auto**, **Single GPU** (pick which one), **Multi-GPU** (split one model across both GPUs), **CPU**.
- Multi-GPU must run ONE model with layers on BOTH the 4070 and the RX 9060 XT at the same time. The CUDA build cannot see AMD, so split mode must use the Vulkan build (or another approach you justify, e.g. llama.cpp RPC). Allow setting the split ratio (default proportional to VRAM) and show how many layers/GB landed on each GPU.
- If only one GPU is available (eGPU unplugged), Multi-GPU falls back to that single GPU automatically with a visible notice — never fail just because the second GPU is missing.
- Keep: separate models on separate GPUs at the same time, crash containment per model, cleanup of server processes when the backend dies.
- Acceptance: on real hardware with the eGPU connected, one model shows VRAM in use on both GPUs and generates correctly; with it disconnected, the same choice loads on the 4070 alone. Add automated tests (the existing `backend/tests/fake_llama_server.py` pattern) for every rule.

## 3. Turn the app into a local coding agent (like Claude Code / Codex)
The app should work as an autonomous coding agent running on local models:
- **System prompts, injected automatically:** a base agent system prompt (role, how to plan, how to use tools, how to edit safely, be concise, verify work); the tool catalogue with schemas; environment context (OS, shell = PowerShell, current workspace path, date, git branch/status, top-level file tree); and per-project instructions read from a file in the workspace (e.g. `AGENTS.md`, also accept `CLAUDE.md`). Keep prompts in versioned files under `backend/app/agents/prompts/`, editable from Settings, with per-model variants for models that need different tool-call formats.
- **Workspace:** the user opens a project folder; the agent works inside it, respecting the existing permission system and workspace boundaries.
- **Tools:** read file, write file, precise edit (string replace / patch) with diffs, list/glob, grep, run shell/PowerShell commands (approval rules + timeouts already exist — reuse them), git status/diff/log/commit, opt-in web search/fetch, and a todo/plan tool. Reuse and extend `backend/app/tools/` and `agents/orchestrator.py`.
- **Agent loop:** multi-step, streams reasoning, tool calls and results live; stops for approval on risky actions; has a step budget; can be interrupted; recovers from tool errors. Use llama-server's native OpenAI tool calling (`--jinja` + `tools`) when the model's chat template supports it, with a robust text-protocol fallback for models that don't. Strip/handle `<think>` blocks correctly.
- **Modes:** Ask (read-only), Plan (proposes a plan, no edits), Agent (edits with approvals), Auto-approve for workspace edits (still never auto-approve destructive commands).
- **Safety and control:** diff review with accept/reject per file, checkpoints/undo of agent edits, an audit log, context management (token budget from the model's real context, summarise/compact long sessions).
- **Sessions:** persistent sessions per workspace; slash commands (e.g. /clear, /compact, /model, /init to generate an AGENTS.md).
- The current models (DeepSeek-R1 1.5B) are weak at tool use. If the network allows, download ONE small tool-capable coding model in GGUF (e.g. Qwen2.5-Coder-7B-Instruct Q4_K_M, under ~6 GB) into `models\` to test the agent end to end. Do not download anything larger.
- Acceptance: in a scratch workspace, the agent can be asked to "create a small Python CLI with tests, run the tests, and fix failures", and it completes the task with visible tool calls, diffs and approvals on a local model.

## 4. State-of-the-art UI
You may throw away the current UI and build a new one from scratch (keep React + TypeScript + Vite + Tailwind unless you have a strong reason; everything must work offline — no CDN fonts or scripts at runtime). Aim for the quality of Claude Code desktop / Cursor / Codex:
- Sessions sidebar, workspace picker with file tree, main agent transcript with collapsible reasoning, tool-call cards (command + output, file edits as syntax-highlighted diffs with accept/reject), approval prompts inline, todo/plan panel, terminal output view.
- Status bar: active model, which GPU(s) it is on, live VRAM per GPU, tokens/sec, context used.
- Model & GPU manager: devices, modes from section 2, load/unload, logs.
- Command palette (Ctrl+K), keyboard-first, dark/light themes, smooth streaming, responsive, accessible, clear empty/loading/error states.
- Keep it running in both Electron (`scripts\start.ps1`) and the browser (`scripts\dev-backend.ps1` → http://127.0.0.1:8765).

## Engineering rules
- Backend stays localhost-only and offline by default; model output is never treated as user approval.
- Every behaviour gets tests. Keep the backend suite green (`cd backend; python -m pytest` — currently 139 tests). Tests must not write to the real `data\` folder (the current suite writes to `data\logs\backend.log`; fix that).
- UI: `npx tsc --noEmit` and `npm run build` clean; take Playwright screenshots of the main screens and check them yourself.
- Real-hardware checks for sections 1 and 2 are mandatory; report actual numbers (VRAM per GPU, tok/s) in `docs/DECISIONS.md`.
- Clear, actionable user-facing error messages. Don't claim anything works unless you ran it.

## When you finish
Update `README.md` and `docs/AGENT_HANDOFF_MULTI_GPU.md` (or replace it with a new handoff doc), and write `docs/REPORT.md` with: what was built, how to run it, what was verified on real hardware (with measurements), what is still missing or unverified, and the next steps. Commit everything.
