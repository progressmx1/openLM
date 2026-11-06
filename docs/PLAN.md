# Implementation Plan — Local-First AI Desktop Assistant

## Environment discovered (2026-10-30)

| Item | Finding |
|------|---------|
| OS | Windows (build 26300), AMD64 |
| CPU | Intel Core Ultra (Family 6 Model 170), 22 logical processors |
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GB VRAM, driver 621.01 (CUDA-capable) |
| Disk | 851.5 GB total, ~12.8 GB free |
| Python | 3.10.0 (`C:\Users\progr\AppData\Local\Programs\Python\Python310\python.exe`) |
| llama-cpp-python | 0.3.36 — **CPU-only build** (`llama_supports_gpu_offload=False`) |
| FastAPI / pydantic / uvicorn | 0.115.6 / 2.10.4 / 0.34.0 (installed) |
| pytest / httpx | 8.3.4 / 0.28.1 (installed) |
| pymupdf / python-docx | **not installed** (needed for Milestone 3) |
| Node / npm / git | 24.14.1 / 11.11.0 / 2.54.0 |

### Models present (all GGUF, same architecture)

- `models/DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf` (3.56 GB)
- `models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf` (1.89 GB)
- `models/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K_L.gguf` (1.58 GB)

DeepSeek-R1-Distill-Qwen-1.5B is a **Qwen2/chatml reasoning model without native tool
calling**; the orchestrator must therefore validate any tool call it emits (Milestone 2+).

## Material deviations from the reference architecture

1. **Python 3.10** is used because it is the interpreter already installed and it is fully
   compatible with the pinned `llama-cpp-python 0.3.36` build (the reference asked for 3.11+).
2. **SQLite is accessed via the stdlib `sqlite3` module** with an explicit schema-version +
   idempotent migration layer, rather than SQLAlchemy/Alembic. This keeps the backend
   dependency-light and avoids an ORM dependency; the migration strategy is documented in
   `backend/app/storage/db.py`.
3. **GPU offload is detected, not assumed.** The installed `llama-cpp-python` wheel is CPU-only,
   so `n_gpu_layers` defaults to `0` on this machine even though an RTX 4070 is present. The
   README documents how to reinstall a CUDA wheel to enable `n_gpu_layers=-1`.
4. **The desktop shell is Electron, but the frontend is also served by the backend** so the app is
   runnable and testable in a normal browser during development and as a fallback.

## Proposed structure

```
offline_models/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routers
│   │   ├── core/           # paths, settings, logging, diagnostics
│   │   ├── models/         # registry + adapters (llama-cpp first)
│   │   ├── agents/         # orchestrator + capability profiles (M2+)
│   │   ├── tools/          # tool registry + document/fs/web/terminal/app tools (M2+)
│   │   ├── security/       # permissions, workspace boundaries, audit (M2)
│   │   ├── storage/        # SQLite persistence
│   │   └── main.py         # FastAPI app factory
│   ├── tests/
│   └── requirements.txt
├── desktop/
│   ├── electron/           # Electron main + preload
│   └── frontend/           # React + TypeScript + Vite + Tailwind
├── data/                   # runtime SQLite + logs (gitignored)
├── models/                 # user GGUF files (gitignored, never committed)
├── scripts/                # launch helpers
├── docs/
└── README.md
```

## Milestones

1. **Foundation + working local chat** — SQLite persistence, GGUF discovery, llama-cpp adapter,
   model selector/load/unload, streaming chat, stop, conversations, diagnostics. *Done.*
2. **Secure tool framework** — tool registry, schemas, permissions, approval UI, audit log,
   workspace-aware filesystem tools, controlled command execution, open/reveal. *Done.*
3. **PDF & DOCX workspace** — read/create/edit/export documents, attachments, references, OCR detect. *Done.*
4. **Internet + coding tools** — opt-in web search/page extraction, coding workspace, diffs, run tests. *Done.*
5. **Advanced** — context management, optional local RAG, multi-model/multi-agent, packaging. *Done (core); packaging deferred.*
