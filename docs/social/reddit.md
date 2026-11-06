# Reddit post (r/LocalLLaMA)

**Title:** I built a local coding agent + chat app on llama.cpp that proves which GPU your model is actually on (and can fall back to Claude/GPT/DeepSeek APIs when you want)

---

I wanted something like Claude Code / Codex that runs on my own laptop (RTX 4070 8 GB, plus an RX 9060 XT eGPU), so I built one. It's a desktop app (Electron or plain browser) with a FastAPI backend and one `llama-server` process per model.

**What it does**

- **Coding agent** with Ask / Plan / Agent / Auto modes. It reads and edits files, runs PowerShell commands and tests, and does git. Every edit is shown as a diff, and in Agent mode it waits for Enter/Esc before applying. Auto mode runs on its own, and destructive commands (`rm -rf`, `git push --force`, `reset --hard` and similar) still ask first. You can keep or revert changes per file, or undo a whole turn.
- **Chat mode** with an optional Web toggle. The model can `web_search` / `web_fetch` and cite the pages it used. Web is off by default.
- **GPU placement you can verify.** Pick Auto / Single GPU / Multi-GPU (Vulkan layer split across NVIDIA + AMD) / CPU. The app parses llama.cpp's own offload log and reads per-process VRAM from Windows GPU counters. So instead of guessing, you get "29/29 layers on RTX 4070, 2.9 GB weights + KV". This started because Task Manager made it look like my model was on the Intel iGPU. It wasn't.
- **Optional cloud models.** Paste an OpenAI / Anthropic / DeepSeek key (or any OpenAI-compatible endpoint) and those models sit next to your GGUFs in the same picker. Keys are encrypted with Windows DPAPI. Local stays the default.

**Things I learned making a 7B model act like an agent**

- Qwen2.5-Coder-7B (Q4_K_M, ~25–31 tok/s fully on the 4070) almost never emitted real tool calls, even with `--jinja`. What fixed it was **JSON-schema-constrained steps**: llama.cpp compiles the schema to a grammar, so every step is a valid tool call or a `respond`. A short `thought` field before the call helped noticeably.
- Small models love to stop and say "Next, I will…". In Auto mode the agent detects unfinished work (open todos, a failed last command, "should I…?") and pushes the model back to work. Until it actually edits or runs something, `respond` is removed from the grammar.
- Small models also copy their own previous status messages, so duplicate status replies get dropped from the context.
- In chat, the 7B model made up a "Tool result:" and a fake release date. Now each step is a grammar-forced choice (search / fetch / answer), followed by a normal streamed answer from the real results.

**Honest limits**

- Simple tasks (fix a failing test, add a function, write a small script) work end to end with the 7B model. Bigger multi-file tasks are limited by the model, and that's partly why I added the API option.
- Multi-GPU split across the 4070 + RX 9060 XT is implemented but not yet tested on the real eGPU. I tested the split mechanism with 4070 + iGPU.
- Windows-only for now (PowerShell, DPAPI, Windows GPU counters).

Stack: Python/FastAPI, llama.cpp b11960 (CUDA + Vulkan builds), React + TypeScript + Tailwind, SQLite. ~280 tests, and none need a GPU (there's a fake llama-server that prints the real log format).

Screenshots below. Happy to answer questions about the constrained-decoding approach or the GPU verification.

---

*Suggested image order:* 01 coding agent (edit approval) → 02 GPU placement → 03 chat with web lookup → 04 cloud models → 05 chat with Claude API.
