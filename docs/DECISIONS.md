# Decisions log

Newest entries are appended at the end of each section. Every measurement here was taken
on the real machine (Acer PHN14-51, RTX 4070 Laptop 8 GB, Intel Arc iGPU, RX 9060 XT eGPU
when connected), llama.cpp build b11960.

---

## D1 — GPU placement investigation (2026-11-02)

**Symptom reported:** "I select the RTX 4070, but Task Manager shows the memory going to the
Intel GPU." The app had launched
`cuda\llama-server.exe -m …-f16.gguf --device CUDA0 -c 28416 -b 512` (no `-ngl`).

**Hardware state during the test:** eGPU *not* connected. `--list-devices`:
- CUDA build: `CUDA0: NVIDIA GeForce RTX 4070 Laptop GPU (8187 MiB, 7068 MiB free)`
- Vulkan build: `Vulkan0: NVIDIA GeForce RTX 4070 Laptop GPU (7948 MiB)`,
  `Vulkan1: Intel(R) Arc(TM) Graphics (9045 MiB)` — note the iGPU reports 9 GB because
  its "VRAM" is shared system RAM.

**Adapter LUIDs (from DXGI `EnumAdapters1`, used to decode the Windows GPU counters):**

| LUID | Adapter | DXGI dedicated |
|---|---|---|
| `0x00015117` | NVIDIA GeForce RTX 4070 Laptop GPU | 7948 MiB |
| `0x00014c79` | Intel(R) Arc(TM) Graphics | 128 MiB (BIOS carve-out) |
| `0x00015089` | Microsoft Basic Render Driver | 0 |
| `0x00017b51` | (in perf counters only; not enumerated by DXGI — stale entry, likely the unplugged eGPU) | – |

**Measurements — the exact same command, re-run with `-lv 4`:**

| | No model | f16 model loaded on CUDA0 |
|---|---|---|
| RTX 4070 adapter dedicated usage | 379 MiB | **4,356 MiB** (+3,977) |
| llama-server process, dedicated on 4070 LUID | – | **3,951 MiB** |
| llama-server process, shared on 4070 LUID | – | 120 MiB |
| llama-server process, any usage on Intel LUID | – | **none (no counter instance)** |
| Intel iGPU adapter shared usage | 1,737 MiB | 1,709 MiB (no growth) |
| `nvidia-smi` total used | 366 MiB | 4,325 MiB |

Verbose server log (these lines are *hidden* at the default verbosity 3 the app used):
```
llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 4070 Laptop GPU) - 7068 MiB free
common_params_fit_impl: projected to use 3813 MiB of device memory vs. 7068 MiB free … no changes needed
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   445.12 MiB
load_tensors:        CUDA0 model buffer size =  2944.68 MiB
llama_kv_cache:      CUDA0 KV buffer size =   777.00 MiB
sched_reserve:      CUDA0 compute buffer size =    92.26 MiB
```
Speed: prompt 2,023 tok/s (1,508-token prompt), generation 36–52 tok/s. CPU-only f16
1.5B on this CPU would be several times slower; these numbers are GPU numbers.

The top users of the Intel iGPU's shared memory with no model loaded were `dwm` (666 MiB),
GalleryApp, PhoneExperienceHost, explorer, the Claude/Edge/WebView2 apps — i.e. the
desktop compositor and UI apps, because the iGPU drives the display (Optimus).

**Conclusion — cause (d), plus a real visibility bug and two latent risks:**
1. The model *was* fully on the RTX 4070 (29/29 layers, ~3.9 GB dedicated). Nothing went to
   the Intel GPU. Task Manager's iGPU "Shared GPU memory" (~1.7 GB) is the desktop/Electron
   compositor, and it did not grow. On this laptop Task Manager's default graphs show the
   3D/Copy/Video engines; CUDA work shows only under the "Cuda"/"Compute_0" engine graph,
   so the 4070 can look idle while it is doing all the work.
2. **Visibility bug (real):** at llama-server's default verbosity the offload and buffer
   lines are not logged, and the app showed no placement proof at all. → Fixed by running
   llama-server with `-lv 4`, parsing those lines, and showing "N/N layers on <GPU>" plus
   live per-process VRAM per adapter from the Windows counters.
3. **Latent risk (c):** with the eGPU unplugged the Vulkan indices shift
   (`Vulkan1` was the RTX 4070 with the eGPU attached, and is the *Intel iGPU* without it).
   A key-only request such as `vulkan:Vulkan1` would silently land on the iGPU. → Fixed:
   a device is always resolved by name, and an integrated GPU is used only when the
   request names that integrated GPU explicitly.
4. **Latent risk (a):** relying on `--fit` with no `-ngl` means that, for a model that is
   slightly too large, llama.cpp silently keeps layers on the CPU, and a saved
   `n_gpu_layers: 0` was ambiguous (it meant "auto" in llama-server but "CPU" in
   llama-cpp-python). → Fixed: when a GPU is chosen we pass `-ngl all` (or the explicit
   positive cap the user set); `0`/`-1` never mean CPU — CPU is its own mode.
5. `CPU_Mapped 445 MiB` is the token-embedding table, which llama.cpp keeps on the host by
   design; it is file-backed (mmap) and is not GPU memory. Whether to disable mmap was
   measured, see D2 (answer: no).

nvidia-smi on Windows/WDDM reports per-process `used_memory` as `[N/A]` (verified), so
per-process VRAM must come from the Windows `GPU Process Memory` counters, which work for
every vendor; nvidia-smi is used only for whole-GPU totals/utilisation.

---

## D2 — Keep mmap (do not pass `--load-mode none`) (2026-11-02)

b11960 replaced `--no-mmap` with `--load-mode {auto,none,mmap,mlock}`. The brief suggested
disabling mmap when fully offloaded so weights are not counted as shared memory. Measured on
the RTX 4070, f16 model, ctx 28416, all 29 layers on CUDA0, via `scripts/verify_gpu.py`:

| load mode | llama-server dedicated (4070) | llama-server **shared** (4070) | gen tok/s |
|---|---|---|---|
| mmap (default) | 3,951 MiB | **120 MiB** | 37.1 |
| none | 3,952 MiB | **566 MiB** | 37.4 |

With `none`, host-side tensors (token embeddings, output staging) go into pinned
`CUDA_Host` memory, which Windows reports as *shared GPU memory*. So `auto` passes no flag
(llama.cpp's default = mmap); `load_mode` stays available per model for users who want it.

## D3 — Multi-GPU = Vulkan `--split-mode layer`, not RPC (2026-11-02)

**Choice:** one model across the RTX 4070 and RX 9060 XT runs in a *single* Vulkan
llama-server with `--device VulkanA,VulkanB --split-mode layer --tensor-split a,b -ngl all`.

**Why not CUDA + RPC:** llama.cpp RPC (`ggml-rpc-server` per GPU, CUDA for NVIDIA + Vulkan
for AMD) would let the 4070 use CUDA kernels, but it adds a second process per GPU, a TCP
hop for every layer boundary per token, more failure modes (the RPC server dies separately
when the eGPU is unplugged), and no crash containment benefit. Layer-split over one Vulkan
process keeps one process per model (the existing crash/cleanup model), and Vulkan on the
4070 measured 62 tok/s vs 60 tok/s for CUDA on Q8_0 — no meaningful speed loss for
generation. RPC remains a possible future option for very large models.

**Rules implemented (`app/models/devices.py`, tested in `tests/test_gpu_modes.py`):**
- Modes per model: `auto`, `single` (= `gpu`, one chosen card), `multi` (= `split`), `cpu`.
- Multi uses the one runtime that sees all chosen cards; if the user picked "RTX 4070 (CUDA)"
  + "RX 9060 XT", the pick is remapped onto the Vulkan entries of the same physical cards.
- Default ratio = proportional to each card's total VRAM; custom ratio allowed.
- If only one GPU is available (eGPU unplugged), Multi falls back to that GPU **on its
  fastest runtime** (CUDA for the 4070) with a visible note; no GPU → CPU with a note.
- The Intel iGPU is never chosen by auto/multi; a key-only request whose index now points at
  the iGPU (index shift) is treated as "GPU not available" and re-placed, with a note.
- Layers per GPU are computed with llama.cpp's own assignment rule (verified against the
  debug log: 3:1 → 22/7 layers); MiB per GPU are the measured buffer sizes from the log.

**Real-hardware results (eGPU not connected during this session):**

| Scenario | Command (relevant flags) | Result |
|---|---|---|
| Single RTX 4070, f16, ctx 28416, saved `n_gpu_layers=0` | `cuda --device CUDA0 -ngl all` | 29/29 layers on 4070; process 3,951 MiB dedicated on 4070; adapter 379→4,330 MiB; iGPU shared 1,685→1,673 (flat); 37 tok/s gen, 352 tok/s prompt (16-token prompt) |
| Multi requested, eGPU absent, Q8_0 | `cuda --device CUDA0 -ngl all` | Fell back to 4070 with note "Multi-GPU fell back to NVIDIA GeForce RTX 4070 Laptop GPU alone (only one GPU is available)"; 29/29 layers; 2,001 MiB; 59.6 tok/s |
| Split mechanism on two real GPUs, Q8_0, 3:1 (4070 + iGPU named explicitly, as the only second GPU present) | `vulkan --device Vulkan0,Vulkan1 --split-mode layer --tensor-split 3,1 -ngl all` | 22 layers / 1,349 MiB on 4070 (counter: 1,363 MiB dedicated), 7 layers / 672 MiB on iGPU (counter: 692 MiB shared); correct output; 38.8 tok/s |

**Not verified:** the RX 9060 XT itself (not connected). To verify when it is plugged in:
`python scripts/verify_gpu.py --model models\DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf --mode multi`
(expect two `per_device` entries, the 9060 XT and the 4070, both with `model_mib > 0`, and
`llama_server_vram_per_adapter` listing both adapters).

---

## D4 — Coding-agent architecture (2026-11-02)

- **New package `backend/app/agents/coding/`** (service, tools, protocol, prompts, environment,
  store) instead of extending the old one-shot `orchestrator.py`, which stays for the legacy
  `/api/chat/agent` endpoint. The old tool registry is reused where it fits (destructive-command
  patterns from `tools/terminal`, web search/fetch, workspace canonicalisation, audit log).
- **One background thread per turn**, events appended to SQLite (`agent_events`) and pushed to
  an in-memory hub; the UI loads history over REST and follows live events over SSE
  (`/api/agent/sessions/{id}/stream?after=seq`). Token deltas are live-only (not persisted); the
  final text is persisted as one `assistant` event.
- **Model-facing history is separate** (`agent_messages`) so `/clear` and `/compact` change what
  the model sees without deleting the transcript.
- **Approvals:** only `POST /decision` (a user click) approves; model text never does. Mode
  policy: Ask/Plan get read-only tools only; Agent asks for every edit and command; Auto applies
  workspace edits immediately (with checkpoints) but still asks for commands unless the user
  chose "Always allow `<prefix>`" for that session; destructive commands (rm -rf, Remove-Item
  -Recurse, git push --force, git reset --hard, format, iwr|iex, …) always ask, even when
  allow-listed. Compound commands (`;`, `|`, `&&`, redirects) are never allow-listable.
- **Edits are prepare → approve → apply.** The diff the user approves is exactly what gets
  written; if the file changed in between, the apply is refused. Every applied edit stores
  before/after content (`agent_checkpoints`) → per-file keep/revert and "undo last turn"; a
  revert is refused if the user changed the file after the agent did.
- **Shell = Windows PowerShell 5.1** (`powershell.exe -NoProfile -NonInteractive`), UTF-8 output
  forced, output streamed live, process tree killed on timeout/stop. `-ExecutionPolicy Bypass`
  is deliberately *not* passed (it would weaken a security setting).
- **Context:** budget = the model's real `n_ctx` (from `/props`) minus the reply budget; old tool
  outputs are shortened first, then older turns are summarised by the model (`compact.md`).
- **Prompts** are versioned files in `backend/app/agents/prompts/` (`system.md`, `modes.md`,
  `tools_native.md`, `tools_text.md`, `tools_json.md`, `compact.md`, `init.md`,
  `variants.json`); Settings → Prompts writes overrides to `<data>/prompts/`.

## D5 — Making a 7B local model actually use tools (2026-11-02)

Measured with Qwen2.5-Coder-7B-Instruct Q4_K_M on the RTX 4070 (`scripts/agent_e2e.py`):

1. Its chat template supports tools and llama-server renders them correctly
   (`/apply-template` shows the `<tools>` block and `<tool_call>` instructions), but the model
   answered agent tasks with a Markdown plan + pasted code and **never** produced a tool call
   (runs 1–3, 0 tool calls).
2. When it did try, it used improvised formats llama.cpp can't parse, e.g.
   `<function name="write_file" arguments='{…}'/>` and `<{"name": …}>`. → `rescue_tool_calls()`
   recovers those (and Qwen3-style `<function=…><parameter=…>`), only for known tool names.
3. `tool_choice: "required"` is **not enforced** for this template in b11960 (still free text).
4. `response_format: {"type": "json_schema"}` **is** enforced (grammar): a valid
   `{"name", "arguments"}` object every time. → **Escalation:** a turn starts with normal
   (native) tool calling; if the model replies with a plan/code block and no tool call, that
   reply is shown but dropped from the context, and the rest of the turn runs with
   JSON-schema-constrained steps: `oneOf` over the mode's tools plus `respond` (finish), each
   with a leading `thought` field (ReAct-style scratchpad, shown as reasoning). Variants can
   force this from the first step (`"tool_format": "json"`).
5. Loop hygiene learned from the runs: edit results now include the edited lines with numbers;
   an identical edit/write is applied at most once per turn; repeated identical calls are
   refused, the temperature is raised while looping, and 3 refused steps in a row end the turn
   (`stuck`) instead of burning the step budget; `edit_file` falls back to a whitespace-tolerant
   unique match (blank lines / trailing spaces / shifted indentation) and says so; before
   finishing in Agent/Auto mode the model is reminded once if its last command failed or todo
   items are open.

## D6 — .gitignore bug (2026-11-02)

The original `.gitignore` used `models/`, which also ignored `backend/app/models/` (all runtime /
placement code) and `desktop/frontend/src/features/models/`; `*.log` ignored the log fixtures.
The baseline commit therefore lacked those files. Patterns are now anchored (`/models/`,
`/data/`, `/runtimes/`) and commit `44a13da` adds the missing source.

## D7 — Later loop fixes from real runs (2026-11-02)

- `run_command` has no `cwd` argument any more: Qwen2.5-Coder repeatedly set `cwd: "tests"` and
  then ran `pytest tests` (path not found), concluded there were no tests, and stopped. Commands
  always run in the workspace root (a subfolder is still reachable with `Set-Location x; …`,
  which is never allow-listed). After this change the UI bug-fix session succeeded first try.
- No-op writes (`write_file` with identical content) are refused.
- After every edit of a `.py`/`.json` file the result includes a syntax diagnostic
  (`WARNING: … now has a syntax error: line N: …`). A run ended with a `try:` without `except`
  in `cli.py`; the model never saw the problem until the test collection error.
- The e2e harness verifies with plain `python -m pytest` (a generated CLI that parses
  `sys.argv` fails under `-q`, which is a harness artifact, not an agent failure).

## D8 — Auto mode is fully autonomous (2026-11-03)

User feedback: in Auto the agent still asked for each new command type and repeatedly ended a
turn mid-project ("The next steps are to implement…", "Please choose an option and proceed"),
so a long build needed "proceed" typed again and again. In the user's WaferLab session the same
status message came back word for word after each "proceed".

- Auto now runs every non-destructive command without asking. Destructive commands
  (`Remove-Item -Recurse`, `rm -rf`, `git push --force`, `git reset --hard`, `format`, `iwr|iex`, …)
  still ask — the one line kept from the original brief. Agent mode is unchanged (asks; per-session
  "Always allow").
- When the model ends a turn while work is visibly left — open todo items, a failed last command,
  text like "next step(s)…", "still need…", an exact repeat of its previous final message, or (Auto
  only) asking the user to choose — the agent sends it back to work automatically: up to 12 times
  per message in Auto, 3 in Agent, never in Ask/Plan. Bounded so it can't loop forever.
- Reasoning-only replies continue automatically (up to 3 in a row) — this part was already in the
  working tree (uncommitted) when I picked the task up, along with Auto as the default mode; both kept.
- Auto prompt: work until done without progress reports; pick sensible options yourself.
- Auto step budget 150 (Agent/Ask/Plan 30).

Real-model check (Qwen2.5-Coder-7B Q4_K_M, RTX 4070, "textkit" package task with 3 modules + CLI +
tests, Auto mode): run 1 — 43 steps, 0 approval prompts, continued automatically 4×, then ended
`stuck` (empty `old_string` edits; fixed: such an edit on a new/empty file now writes it, and in
Auto a looping model is redirected before giving up). Run 2 — 44 steps, 0 approval prompts, used
all 12 automatic continuations, context compacted once, ended with 2 of 6 tests passing. The
agent no longer stops waiting for "proceed"; finishing such tasks correctly is now limited by the
7B model's coding ability.

## D9 — Chat side, web lookups, and long-turn compaction (2026-11-03)

- **Chat** is a session kind of the same engine (`mode="chat"`, no workspace): streaming,
  reasoning, models and sessions are shared with Code; its tools are only `web_search` /
  `web_fetch` (when Web is on); prompt `chat.md`. Chats and code sessions are listed separately
  and cannot be converted into each other.
- **Grounded lookups for weak tool callers.** Qwen2.5-Coder-7B, asked a question needing current
  info, wrote a fake "Tool result:" in prose and invented a release (v0.1.2, 2023). Now, in chat
  with Web on and a `json`-variant model, each step first makes a JSON-schema-constrained choice
  (`web_search` / `web_fetch` / `answer`, ≤4 lookups, identical lookups refused); the final answer
  is a normal streamed call with an explicit "answer from the results, cite pages" instruction.
  Real run: searched → fetched github.com/ggml-org/llama.cpp/releases/tag/b12032 → answered with
  the correct tag, date and link; some listed details were not on the page (7B embellishment).
- **Web search** no longer needs `cssselect` (not in requirements; the WaferLab session crashed on
  it and the agent pip-installed it): XPath parsing, DuckDuckGo redirect links decoded to the real
  URL, page text taken from `<main>`/`<article>` to skip site chrome.
- **Compaction inside one long turn** was a no-op that ran on almost every step (it only
  summarised earlier turns), so long Auto runs kept silently dropping their oldest messages. Now
  it summarises the middle, keeps the task message and the newest messages (≈ half the budget),
  never drops the summary or the task, and only announces itself when it actually summarised.

## D10 — Paid cloud models via API (2026-11-03)

- **Optional, per model.** A cloud model is registered as `cloud://<provider>/<model>` (OpenAI,
  Anthropic, DeepSeek, or any OpenAI-compatible endpoint such as OpenRouter or a LAN server) and
  appears next to local models in Chat and Code. A `ProviderRouter` sends `cloud://` models to
  `CloudAdapter` and everything else to the llama.cpp adapter. Cloud models take no GPU, so
  "loading" one never unloads a local model. A cloud model is ready as soon as its provider has a key.
- **Subscriptions are not API keys.** ChatGPT Plus and Claude Pro/Max cannot be used by other
  apps; you need API keys (platform.openai.com, console.anthropic.com, platform.deepseek.com),
  which are billed per token.
- **Keys** are encrypted with Windows DPAPI (current user only) in `data\secrets.bin`. They are
  never returned to the UI (only the masked `sk-…abcd` form is), and are not stored in settings
  or the database.
- **OpenAI uses the Responses API** (`/v1/responses`). With a real key, gpt-6-luna rejected
  function tools in `/chat/completions` while reasoning was on (400: "use /v1/responses or set
  reasoning_effort to 'none'"). Turning reasoning off would weaken the model, so OpenAI requests
  now go to `/v1/responses`. The request uses `store: false`, `reasoning: {effort, summary:
  "auto"}` and non-strict function tools. Reasoning summaries stream into the reasoning panel, and
  tool-call arguments are taken from deltas, or from the `.done` events when those are the only
  place they appear. A call that never got a result gets "(not run)" as its result. A rejected
  setting is dropped and the request retried: reasoning summaries (they need a verified
  organisation), temperature, or an unsupported effort level. DeepSeek and custom endpoints stay
  on `/chat/completions`.
- **OpenAI-compatible providers** reuse the llama-server SSE parser (now `openai_stream.py`), so
  native tool calls work unchanged. OpenAI uses `max_completion_tokens`, and reasoning models
  (o-series, gpt-5) get no temperature. DeepSeek's `reasoning_content` is shown as reasoning.
- **Claude** goes through the official `anthropic` SDK with streaming, adaptive thinking
  (summarised and shown in the reasoning panel), `effort` set to high for code and medium for
  chat, and no sampling parameters.
  - Tools use `eager_input_streaming`.
  - On models that support them, server-side fallbacks are on by default; a request is retried
    without them if the account rejects them.
  - Thinking blocks are **not sent back**: the agent edits its own history (compaction, shortened
    tool output), and replayed thinking must match the original exactly, so dropping it is the
    safe option.
  - The system prompt changes every turn (git status, date), so prompt caching gains little and
    is not used.
- **The agent treats every cloud model as native-tools** (no text protocol, no constrained JSON).
- **Privacy:** with a cloud model, prompts, file contents the agent reads and command output go
  to the provider. The UI says so, and every cloud model is labelled "· <Provider> API". Local
  models still work fully offline. The Web toggle only controls web search and fetch.
- **Verified:** 13 unit tests (MockTransport for OpenAI/DeepSeek, a fake client for Claude:
  request shape, no sampling params, no thinking replayed, tool-use mapping, fallback retry,
  encrypted key file). In the UI, a dummy DeepSeek key was saved, `deepseek-chat` was added and
  picked in Chat, and the real DeepSeek API answered "rejected the API key (401)", shown as a
  readable error. **Not verified:** a successful answer from a paid model, because no real key
  was available.
