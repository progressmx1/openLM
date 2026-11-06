# Coding Agent Prompt: Build a Local-First AI Desktop Assistant

You are a senior full-stack, desktop, and AI-infrastructure engineer. Build a **production-quality local-first AI desktop application** that turns my downloaded local language models into a fully integrated chat and computer-assistance environment.

The application must have a polished graphical user interface (GUI), work with local models while offline, and provide a controlled set of tools for documents, files, internet research, and software development. It should feel like a capable desktop AI workspace—not a thin chat demo.

**Your job is to inspect the existing project, make an implementation plan, and then build and test the application. Do not stop after writing a plan, generating mockups, or creating placeholder screens.** Implement working functionality end to end, document anything that cannot be completed, and never represent a stub as a working feature.

---

## 1. Main Goal

Create a desktop application where I can:

1. Select and chat with any **supported local model** installed on my computer.
2. Run local models completely offline, without API keys or cloud services.
3. Discover, register, load, unload, and switch between multiple local models.
4. Attach, read, search, summarize, create, and edit PDF and DOCX files.
5. Ask the assistant to create, inspect, edit, run, and organize code projects and files.
6. Give the assistant controlled access to the internet for search and research when enabled.
7. Ask it to execute terminal commands, run scripts, and open files or applications on my PC—with clear permissions and user approval.
8. See what the assistant is doing, which tools it is calling, what files it is accessing, and the actual results of those actions.
9. Keep conversations, settings, model registrations, and task history locally.
10. Optionally coordinate multiple local models in a multi-agent workflow, where available models can review, critique, or collaborate on a task.

The application should be **local-first, offline-capable, secure by default, modular, and extensible**.

## 2. First Steps: Inspect Before Changing Anything

Before implementing:

- Inspect the repository, current files, existing Python environment, downloaded models, and current use of `llama-cpp-python`.
- Identify the operating system and available hardware where possible, including CPU, RAM, GPU, VRAM, CUDA availability, and disk space. Hardware detection must be best-effort and must not crash when a query is unsupported.
- Identify the actual model file formats present. Do not assume every model is GGUF.
- Preserve useful existing code and configuration. Do not overwrite or delete existing model files.
- Check whether the project already has a frontend, backend, database, or working model loader. Reuse sound components rather than rebuilding them unnecessarily.
- Write a short implementation plan and a clear proposed project structure, then proceed with implementation without waiting for approval unless a genuinely blocking decision or destructive change is required.

If the repository is empty, initialize the project using the architecture below. If the repository is already established, adapt the architecture to it and explain material deviations in the README.

## 3. Recommended Technical Architecture

Use a clean separation between the desktop UI, application backend, model runtimes, agent orchestration, and tools.

### Preferred stack

- **Desktop shell:** Electron for the first Windows release, unless the existing project already uses another suitable desktop framework.
- **Frontend:** React + TypeScript.
- **UI:** Tailwind CSS plus a consistent, accessible component library. Use a polished, restrained design with a dark/light theme.
- **Backend:** Python 3.11+.
- **Backend API:** FastAPI, using a localhost-only service or a secure desktop IPC bridge as appropriate.
- **Local inference:** `llama-cpp-python` / `llama.cpp` for GGUF models.
- **Optional additional local runtimes:** adapters for Ollama and OpenAI-compatible local endpoints such as LM Studio, where practical.
- **Persistence:** SQLite for conversations, settings, model registry, task records, permissions, and tool logs.
- **PDF:** PyMuPDF or pypdf for PDF text and page operations; ReportLab or an equivalent library for PDF creation.
- **DOCX:** `python-docx` for document extraction and creation.
- **Scanned PDF OCR:** Optional local OCR integration, such as Tesseract, only when installed and configured.
- **Testing:** pytest for Python, Vitest/React Testing Library for frontend, and Playwright for desktop/UI end-to-end coverage where practical.

Keep the backend modular. Do not implement the whole application in one Python file or one React component.

### Logical architecture

```text
Desktop app (Electron)
└── React + TypeScript UI
    ├── Chat workspace
    ├── Model selector and model manager
    ├── File/document workspace
    ├── Tool activity and approval center
    ├── Coding workspace
    ├── Optional multi-agent workspace
    └── Settings and diagnostics

Application bridge / localhost API
└── Python backend (FastAPI)
    ├── Conversation and context manager
    ├── Model registry
    ├── Model runtime adapters
    │   ├── llama-cpp-python (GGUF)
    │   ├── Ollama (optional)
    │   └── OpenAI-compatible local endpoint (optional)
    ├── Agent orchestrator
    ├── Tool registry and permission manager
    ├── Document tools (PDF/DOCX)
    ├── File system tools
    ├── Web research tools
    ├── Coding and terminal tools
    ├── Application/file opening tools
    ├── Optional multi-agent coordinator
    ├── SQLite persistence
    └── Audit log and diagnostics
```

The model and tool layers must communicate through stable interfaces. The UI must not be coupled directly to `llama-cpp-python` internals.

## 4. Desktop UI and User Experience

Build a professional desktop interface with the following areas.

### 4.1 Main chat screen

- Left sidebar with conversations, search, new chat, rename, pin, archive, and delete.
- Main chat area with user and assistant messages.
- Model selector clearly visible in the chat header.
- Composer with multiline input, send button, stop button, and attachment button.
- Drag-and-drop attachments.
- Streaming responses where supported.
- Markdown rendering, tables, mathematical notation, and syntax-highlighted code.
- Copy response, copy code, regenerate, edit/resend, and retry controls.
- Support multiple conversations and persist them locally.
- Show the selected model, runtime, local/remote status, and whether the app is offline.
- Display model loading and generation state, token counts or speed when available, and context usage when it can be measured reliably.
- Clearly display errors, timeouts, and interrupted tasks. Do not silently discard errors.

### 4.2 Tool activity panel

Show tool activity as it happens. Each tool entry should show:

- Tool name and a plain-language description.
- What the assistant wants to do.
- The relevant file path, URL, or command.
- Permission classification and approval state.
- Running, succeeded, failed, cancelled, or timed-out status.
- Actual tool output, including generated file paths.

Let the user expand or collapse tool details, cancel a running operation, and inspect results. Tool execution must not be hidden inside ordinary assistant text.

### 4.3 File and document workspace

Provide a view of attached files, recently generated files, and user-approved workspace folders. Include actions to preview, open, reveal in File Explorer, and copy the path. Do not imply that a file exists until the backend confirms it.

### 4.4 Settings and diagnostics

Include settings for:

- Model directories and registered model files.
- CPU/GPU inference settings.
- Default generation settings.
- Approved workspace directories.
- Tool permissions and approval preferences.
- Internet access and search configuration.
- Optional external model providers and API keys.
- Theme, chat behavior, data retention, and logs.
- Backend status, installed dependencies, runtime versions, hardware detection, and diagnostic reports.

API keys must be stored using an operating-system secure credential store where possible (for example, Windows Credential Manager or an established secure-storage library), not in plain-text source files or committed configuration.

## 5. Local Model Management and Offline Inference

This is a core feature. Make it reliable and easy to use.

### 5.1 Model discovery and registration

- Let me add one or more folders containing local models.
- Scan selected folders for supported model files.
- Initially support `.gguf` files through `llama-cpp-python`.
- Detect other installed local runtimes and compatible endpoints where feasible, but do not claim arbitrary formats are supported unless a real backend can load them.
- Provide a manual file picker to register a model.
- Show model name, file path, file size, format, quantization and metadata when detectable, runtime, and availability.
- Allow custom display names and descriptions.
- Detect missing files and report them without crashing.
- Never move, rename, modify, or delete source model files as part of discovery.

### 5.2 Model selection and lifecycle

- Allow selection of any registered, compatible model from the chat header.
- Load a selected model on demand and show status/progress.
- Unload a model to free RAM/VRAM.
- Make the active model obvious in the UI.
- Provide understandable errors for unsupported formats, missing dependencies, incompatible GPU builds, insufficient memory, and invalid model files.
- Prevent accidental simultaneous loading of several large models by default. Allow multi-model loading only when the user enables it and the machine has sufficient resources.
- Do not crash the UI if a model fails to load or generate.
- Keep the application responsive while a model loads or generates.

### 5.3 Inference configuration

Provide sensible defaults and advanced per-model settings for:

- Context length (`n_ctx`).
- GPU layers (`n_gpu_layers`).
- CPU threads.
- Batch size and micro-batch size where supported.
- Temperature, top-p, top-k, repeat penalty, and maximum output tokens.
- Seed and stop sequences where supported.
- Chat template / prompt format.

Use the model's correct chat template when available. Do not assume every GGUF model uses the same template or supports native tool calling. Let advanced users override template settings. If a model does not support native tool calling, the orchestration layer may use a controlled structured-output strategy, but must validate every proposed tool call before execution.

GPU settings must be hardware-aware and configurable. `n_gpu_layers=-1` may be offered as an option, but must not be assumed to work on every machine. Explain that memory usage depends on model size, quantization, context length, and offloaded layers.

### 5.4 Model adapter interface

Create a runtime abstraction similar to:

```python
class ModelAdapter:
    def discover_models(self, locations): ...
    def load_model(self, model_id: str, settings: dict): ...
    def unload_model(self, model_id: str): ...
    def chat(self, messages: list, options: dict): ...
    def stream_chat(self, messages: list, options: dict): ...
    def get_model_status(self, model_id: str): ...
    def get_runtime_info(self): ...
```

Implement the `llama-cpp-python` adapter first. Add other adapters only when they can be implemented and tested. A local OpenAI-compatible endpoint is a connection to another local runtime, not proof that the endpoint itself is offline; show that distinction accurately.

### 5.5 Offline behavior

- Local model chat, local file tools, document processing, conversation history, and local coding tasks must work without internet once dependencies are installed.
- Do not make background network calls in offline mode.
- Clearly indicate when a requested feature requires an internet connection or a missing local dependency.
- Do not require cloud authentication for local model use.

## 6. Agent Orchestration and Tool Calling

Implement an agent loop that can interpret a user request, decide whether a tool is needed, request a tool call, validate it, obtain permission when required, execute it, inspect the result, and then respond.

### Requirements

- Use a central tool registry with validated JSON input schemas.
- Never execute raw model-generated shell or Python text directly without routing it through the permission manager and execution layer.
- Validate arguments, paths, file types, and command parameters on the backend—not only in the UI.
- Apply timeouts, cancellation, output limits, and error handling.
- Return structured tool results to the model.
- Prevent infinite tool-call loops with configurable step and time limits.
- Support a maximum number of tool iterations per task and show when the limit is reached.
- Keep the user informed while longer tasks run.
- Let the user stop generation and cancel tools where cancellation is supported.
- Save task and tool execution history locally.

Tool calling varies by model. The application must not assume every model supports the same tool-call protocol. Create a model capability profile and a compatible orchestration fallback. Do not execute malformed or ambiguous tool requests.

## 7. Security, Permissions, and Computer Access

The application will be able to access local files and run commands, so security is a first-class requirement. **Never implement unrestricted, invisible computer control.** The assistant should be capable, but the user remains in control.

### 7.1 Permission categories

Implement these permission levels:

1. **Read-only:** List, search, and read files within approved locations; inspect documents; perform permitted web searches.
2. **Workspace write:** Create or edit files only inside directories explicitly approved by the user.
3. **System action:** Execute commands, install packages, launch applications, or access paths outside approved workspaces. Require explicit approval unless the user has set a narrowly scoped trusted rule.
4. **High-impact action:** Delete or overwrite important files, modify system settings, change permissions, access credential stores, or perform actions with significant external consequences. Always require a separate, explicit confirmation describing the exact action and target.

A model-generated request is never user approval.

### 7.2 Workspace boundaries

- Let the user select trusted project/workspace folders.
- Restrict file creation, editing, and deletion to approved workspaces by default.
- Canonicalize paths and resolve symlinks before checking boundaries to prevent path traversal.
- Block attempts to escape an approved workspace using `..`, symlinks, junctions, or alternate path forms.
- Make access outside approved folders an approval-required action.
- Do not access browser profiles, password stores, private keys, tokens, or other credentials unless a specific user-initiated workflow requires it and the user explicitly authorizes it.

### 7.3 Command execution

Create a controlled command runner that can:

- Run Python scripts and common development commands.
- Use a selected project working directory.
- Capture stdout, stderr, exit code, and elapsed time.
- Stream output to the UI for long-running commands.
- Support timeout and cancellation.
- Show the exact command and working directory before execution.
- Require approval for commands that modify the system, install software, access external services, or operate outside the workspace.
- Use an isolated environment or sandbox for untrusted code when practical.

Do not run every command with administrator privileges. Do not automatically elevate privileges. Never silently run destructive commands such as recursive deletion, disk formatting, registry changes, firewall changes, or system-level configuration edits. Always require a separate confirmation for these high-impact operations.

### 7.4 Opening files and applications

- Provide a tool to open a user-selected or workspace-approved file using the operating system's default application.
- Provide a tool to reveal a file or folder in File Explorer.
- Provide a controlled way to launch common developer applications if installed.
- Show the exact file or application being opened.
- Ask for approval when the action involves an untrusted executable, a downloaded file, or a path outside the approved workspace.
- Do not execute a file merely because it was opened or mentioned in a document.

### 7.5 Untrusted content and prompt injection

Treat content from PDFs, DOCX files, source code, terminal output, websites, and search results as **untrusted data**, not instructions. A document or webpage may contain text telling the assistant to ignore its rules, reveal secrets, or run commands. The agent must not obey such embedded instructions. Tool authorization must come from the user's current request and the permission manager, not from the content being analyzed.

### 7.6 Audit trail

Record, locally:

- Timestamp, conversation/task ID, model ID, tool name, and permission level.
- Relevant paths, URLs, or sanitized command details.
- Approval or rejection decision.
- Execution status, exit code, and concise result.

Do not log API keys, passwords, access tokens, or unnecessary sensitive document contents. Provide a way to inspect and clear local logs.

## 8. PDF and DOCX Capabilities

Create real document tools, not merely instructions for the model to imitate.

### 8.1 PDF tools

Support:

- Read PDF metadata and page count.
- Extract text page by page.
- Search for phrases and return page numbers and relevant excerpts.
- Summarize a whole document or selected pages.
- Extract tables where feasible.
- Create PDFs from text and structured content.
- Merge PDFs and split them into selected pages/files.
- Extract selected pages and rotate pages.
- Add page numbers where feasible.
- Convert extracted content to TXT or DOCX.
- Detect image-only/scanned PDFs and report when OCR is needed.
- Offer local OCR if a supported OCR engine is installed; do not claim OCR was performed if it was not.

### 8.2 DOCX tools

Support:

- Read paragraphs, headings, tables, lists, headers, footers, and basic metadata.
- Create new DOCX documents.
- Add and edit paragraphs, headings, tables, lists, and page breaks.
- Make formatting changes while preserving existing formatting as much as reasonably possible.
- Export to PDF when a local conversion engine is available.
- Report limitations when the document contains unsupported formatting or embedded objects.

### 8.3 Safe editing workflow

- Preserve the original document by default.
- Save edits as a new file/version unless the user explicitly authorizes overwriting.
- Show the output path and offer Open File and Reveal in Folder actions.
- For edits, provide a concise summary of what changed.
- For long documents, process in chunks and use retrieval rather than forcing the entire document into the model's context window.
- When answering questions about a document, cite page numbers or section headings where available.

## 9. File System and Project Tools

Implement validated tools for:

- List directories and files.
- Search file names and, where practical, file contents in approved locations.
- Read text, Markdown, JSON, YAML, CSV, and source-code files.
- Read PDF and DOCX through the document subsystem.
- Create folders and files.
- Write and edit text/code files.
- Copy and move files within approved locations.
- Rename files with approval where appropriate.
- Compare file versions or show diffs before applying code/document edits.
- Save generated artifacts in a chosen output directory.
- Report file size, type, modified date, and exact path.

Do not silently overwrite an existing file. Use a safe default such as a new filename or versioned backup. Deletion must require explicit confirmation.

## 10. Internet and Research Tools

Internet features must be optional and visibly separate from offline/local capabilities.

Provide tools for:

- Search the web using a configurable search provider.
- Open and extract readable webpage content.
- Visit user-provided URLs.
- Retrieve public documentation and technical references.
- Download user-requested files to an approved download/workspace directory, with confirmation for executable or potentially unsafe file types.
- Summarize and compare sources with clickable source links.

Requirements:

- Provide an Internet Access setting and visible online/offline indicator.
- Do not send local files, conversation contents, prompts, or private data to external services unless the user explicitly requests that action.
- Do not silently switch from a local model to a cloud model.
- Clearly label whether a response used local inference, an external model provider, or web data.
- Respect website access controls and reasonable request limits. Do not bypass logins, paywalls, CAPTCHAs, or access restrictions.
- Treat webpage text as untrusted content and defend against prompt injection.
- Store provider API keys securely if a provider requires one. Offer search providers that do not require a key where practical.

## 11. Coding Assistant and Development Tools

Add a coding mode or coding-aware chat workflow that can:

- Inspect a selected project directory.
- Understand project structure and relevant files.
- Read source code and configuration.
- Search symbols and references.
- Explain code and diagnose errors.
- Create new files and edit existing code.
- Show diffs before applying significant changes.
- Run tests, linters, formatters, builds, and user-approved scripts.
- Read compiler/test output and iterate on fixes.
- Use Git status and diff commands where Git is installed.
- Explain which files changed and why.

Do not automatically commit, push, publish, deploy, or upload code. These actions require explicit user authorization. Before making broad changes, inspect the project and propose a concise plan. Keep edits within the selected project unless approved otherwise.

Support common development environments where installed, including Python, Node.js, Git, and C/C++ toolchains. Detect available tools rather than assuming they exist. Provide clear setup guidance when a dependency is missing.

## 12. Optional Multi-Model and Multi-Agent Workflows

Design the system so multiple local models can collaborate without making this a blocker for the first usable release.

Provide an optional multi-model workspace where the user can:

- Select two or more registered models.
- Assign roles such as Planner, Researcher, Coder, Reviewer, and Final Synthesizer.
- Run a sequential workflow, debate/review workflow, or parallel independent analysis.
- Set a maximum number of rounds and token/time budget.
- Inspect each model's contribution and the final synthesis.
- Stop the workflow at any time.

Models may have different context lengths, tool-call capabilities, and resource requirements. The coordinator must account for this and avoid loading more models than available memory can support. Multi-agent mode must not bypass the same tool permissions and approval system used by normal chat.

Implement this after the single-model chat, tool calling, and permission system are working.

## 13. Context, Memory, and Retrieval

Implement practical conversation context management:

- Persist full conversation history locally.
- Allow starting a new chat with a clean context.
- Provide configurable context limits and a visible context indicator where reliable.
- Summarize older conversation turns when needed, while preserving access to original history.
- Keep file/document retrieval scoped to the files attached or approved for the current task.
- Do not automatically index the entire computer.
- Let the user explicitly choose folders to index for local search/RAG.
- Provide controls to inspect and delete indexed sources and stored conversation data.

For RAG, use a modular local embedding/vector-store implementation. Make embedding model selection and storage location visible. RAG must be optional and should work offline when the embedding model is available locally.

## 14. Reliability and Performance

- Run model inference and long-running tools asynchronously so the UI remains responsive.
- Support streaming output and cancellation where technically possible.
- Add sensible timeouts and output-size limits.
- Handle unavailable models, corrupted files, GPU errors, missing dependencies, network failures, and malformed tool calls gracefully.
- Prevent duplicate tool execution when a request is retried.
- Use structured logs with sensitive values redacted.
- Ensure the backend can restart without corrupting the conversation database.
- Avoid keeping large document contents or model weights duplicated in memory unnecessarily.
- Clearly distinguish a failed action from an action that completed with warnings.

## 15. Data and Privacy Defaults

- Keep conversations, settings, logs, and model registry on the user's computer by default.
- Do not upload local data to a cloud service by default.
- Do not enable telemetry by default. If optional telemetry is added, make it opt-in and explain exactly what is collected.
- Provide controls to export and delete conversations and local application data.
- Keep secrets out of logs, prompts, and ordinary settings files.
- Explain what data leaves the computer whenever the user enables an external provider or submits data to an internet service.

## 16. Suggested Project Structure

Adapt this to the repository rather than forcing it blindly:

```text
local-ai-desktop/
├── desktop/
│   ├── electron/
│   └── frontend/
│       ├── src/
│       │   ├── components/
│       │   ├── features/
│       │   │   ├── chat/
│       │   │   ├── models/
│       │   │   ├── files/
│       │   │   ├── tools/
│       │   │   ├── coding/
│       │   │   ├── agents/
│       │   │   └── settings/
│       │   ├── hooks/
│       │   ├── services/
│       │   ├── styles/
│       │   └── types/
│       └── package.json
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   │   ├── adapters/
│   │   │   └── registry.py
│   │   ├── agents/
│   │   ├── tools/
│   │   │   ├── documents/
│   │   │   ├── filesystem/
│   │   │   ├── web/
│   │   │   ├── terminal/
│   │   │   └── applications/
│   │   ├── security/
│   │   ├── storage/
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── data/                 # runtime data; must be gitignored
├── models/               # optional user model directory; do not bundle weights
├── scripts/
├── docs/
├── .gitignore
└── README.md
```

Never commit model weights, user documents, API keys, local databases, logs, or generated private files.

## 17. API and Internal Contracts

Define typed request/response schemas for at least:

- Model discovery, registration, load, unload, and status.
- Chat completion and streaming events.
- Conversation CRUD and message history.
- Tool discovery, tool execution request, approval decision, and execution result.
- File selection and attachment processing.
- Document operations.
- Command execution and cancellation.
- Web search and page extraction.
- System diagnostics and settings.

Use explicit status values and structured errors. Document the API and avoid exposing arbitrary backend functions directly to the frontend.

## 18. Testing Requirements

Write automated tests for the core behavior. At minimum, test:

### Model runtime

- Discover a valid GGUF model in a temporary test directory.
- Handle missing and invalid model paths.
- Load/unload behavior with a mock adapter.
- Model switching and inference errors.
- Streaming and cancellation behavior where supported.

### Permissions and file security

- Access allowed inside a workspace.
- Reject path traversal outside a workspace.
- Reject symlink/junction escapes where relevant.
- Require approval for outside-workspace writes and system commands.
- Require explicit confirmation for delete/overwrite operations.
- Reject malformed tool arguments.

### Documents

- Read a test PDF and report page references.
- Create a PDF and verify it exists and is readable.
- Read and create a DOCX with paragraphs and tables.
- Preserve the source file during default edit workflows.
- Handle scanned PDFs when OCR is not installed by reporting the limitation accurately.

### Agent/tools

- Tool call validation.
- Approval/rejection flow.
- Tool timeout and cancellation.
- Tool failure is returned to the assistant rather than treated as success.
- Prompt-injection text in a test document or webpage cannot grant tool permissions.

### UI and persistence

- New conversation, send message, switch model, and reload conversation.
- Model loading and error states.
- Approval dialog and tool activity display.
- App restart preserves conversations and settings.

Use mocks for tests that must not depend on a large model, a real GPU, external internet, or destructive system operations. Add a separate optional smoke test for a real local GGUF model.

## 19. Build Plan and Priorities

Implement in milestones. Keep the app runnable at the end of each milestone.

### Milestone 1 — Foundation and working local chat

- Inspect current repository and environment.
- Create or stabilize desktop UI and Python backend.
- Implement SQLite persistence.
- Implement model discovery for GGUF files.
- Implement `llama-cpp-python` adapter.
- Build model selector, model loading/unloading, and streaming chat.
- Ensure local chat works offline.

### Milestone 2 — Secure tool framework

- Implement tool registry, schemas, permissions, approval UI, audit logs, cancellation, and timeouts.
- Implement workspace-aware filesystem tools.
- Implement controlled command execution.
- Implement opening/revealing files.
- Add security tests before expanding tool access.

### Milestone 3 — PDF and DOCX workspace

- Implement PDF and DOCX reading, creation, editing, and export tools.
- Add attachments, previews/status, page/section references, and safe output handling.
- Add optional OCR detection/integration.

### Milestone 4 — Internet and coding tools

- Implement opt-in web search and page extraction.
- Implement coding workspace, diffs, test/build execution, and Git inspection.
- Add clear local-versus-network data indicators.

### Milestone 5 — Advanced agent features

- Improve context management and optional local RAG.
- Add multi-model/multi-agent workflows.
- Add advanced runtime integrations if useful.
- Polish diagnostics, recovery, packaging, and documentation.

Do not skip Milestone 2 security work in order to make command execution appear more impressive.

## 20. Acceptance Criteria

The application is not complete until the following are demonstrated:

1. I can launch the desktop app on Windows and see a working GUI.
2. I can register at least two local GGUF models from folders on my computer.
3. I can select a model, load it, chat with it, stream its response, stop generation, unload it, and switch to another model.
4. Local-model chat and core local tools work with the internet disconnected.
5. Conversations and settings persist after restarting the app.
6. I can attach a PDF and DOCX, ask questions about them, and get answers grounded in their actual content.
7. I can ask the assistant to create a DOCX and PDF and verify that the output files exist and open correctly.
8. I can select a project folder and ask the assistant to inspect and edit code; meaningful edits are shown as diffs or clearly summarized.
9. I can ask it to run a test or script, see the exact command, approve it when required, and inspect the actual output and exit code.
10. I can ask it to open a generated file or reveal it in File Explorer.
11. Internet access can be enabled or disabled, and internet use is visible in the UI.
12. The assistant cannot silently execute destructive commands or write outside approved directories.
13. Tool calls, approvals, failures, and outputs are visible and recorded locally.
14. Errors are handled gracefully, and the README explains setup, installation, model registration, GPU configuration, permissions, and troubleshooting.
15. Automated tests cover the critical model, document, permission, and persistence workflows.

## 21. Deliverables

Provide:

- Complete application source code.
- Working desktop UI and Python backend.
- Dependency files with pinned or bounded versions as appropriate.
- `.env.example` only for non-secret configuration; never put real credentials in it.
- Database initialization and migration strategy.
- Automated tests and instructions to run them.
- A detailed README with Windows setup and launch instructions.
- Instructions for installing the correct `llama-cpp-python` build for CPU or the user's available GPU backend.
- Model registration and configuration guide.
- Tool permission and security guide.
- Troubleshooting guide for model loading, VRAM/RAM, chat templates, PDF/DOCX conversion, and missing system dependencies.
- A clear list of implemented features, limitations, and deferred features.

## 22. Working Rules for the Coding Agent

- Make real, testable changes; do not only describe what should be built.
- Start by inspecting the repository and existing downloaded model files. Do not ask me to repeat information that can be discovered locally.
- Preserve existing working code and user data.
- Do not download or replace large model weights unless I explicitly ask.
- Do not install system-wide packages or change system configuration without approval. Prefer a project virtual environment.
- Do not hard-code local absolute paths; make model and workspace directories configurable.
- Do not assume CUDA, NVIDIA, or a specific GPU is present.
- Do not assume every model supports tool calling or uses the same chat template.
- Do not silently use a cloud model as a fallback.
- Do not mark a feature complete until it has been exercised and tested.
- When something cannot be implemented in the current environment, explain the exact blocker and provide the closest safe, working alternative.
- After each milestone, report: files changed, functionality implemented, tests run and results, known limitations, and the command to launch the application.
- Keep the project runnable throughout development.

## Final Instruction

Begin by inspecting the current project and the local model directory. Then implement **Milestone 1** first so I have a working desktop chat with selectable local models. Continue through the remaining milestones in order, keeping the security and permission system in place before enabling powerful computer-control tools. The finished product should be a dependable local AI workspace that I can use offline for everyday chat, document work, research, and software development.
