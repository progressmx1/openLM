"""Optional multi-model coordination.

Runs a bounded workflow across two or more registered models with explicit roles.
Models may have different capabilities; the coordinator accounts for load state and
never bypasses the permission system (tool use, when present, still goes through the
executor). Workflows are sequential, review (draft + critique), or parallel.

In the parallel workflow, agents whose models live on different GPUs genuinely
run at the same time (each model has its own server process and lock); agents
sharing one model are serialized by that model's lock.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Optional

from ..core.settings import SettingsStore
from ..models.registry import ModelRegistry


class MultiAgentCoordinator:
    def __init__(self, model_registry: ModelRegistry, settings_store: SettingsStore) -> None:
        self.models = model_registry
        self.settings_store = settings_store

    def _options(self, overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
        options = self.settings_store.load().inference.model_dump()
        if overrides:
            options.update({k: v for k, v in overrides.items() if v is not None})
        return options

    def _run_agent(self, model_id: str, role: str, context: list[dict[str, Any]], options: dict[str, Any]) -> str:
        messages = [{"role": "system", "content": f"You are the {role} in a multi-agent workflow."}]
        messages += context
        result = self.models.adapter.chat(model_id, messages, options)
        return result["content"]

    def run(
        self,
        task: str,
        agents: list[dict[str, str]],
        workflow: str = "sequential",
        max_rounds: int = 1,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if len(agents) < 2:
            return {"status": "error", "error": "At least two agents are required."}

        opts = self._options(options)
        # Validate load state up front (best-effort; individual turns also guard).
        unloaded = [a["model_id"] for a in agents if not self.models.adapter.is_loaded(a["model_id"])]
        if unloaded:
            return {"status": "error", "error": f"Models not loaded: {unloaded}"}

        steps: list[dict[str, Any]] = []
        deadline = time.monotonic() + 300

        if workflow == "parallel":
            pool = ThreadPoolExecutor(max_workers=len(agents), thread_name_prefix="agent")
            futures = [
                pool.submit(
                    self._run_agent, agent["model_id"], agent["role"], [{"role": "user", "content": task}], opts
                )
                for agent in agents
            ]
            done, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
            pool.shutdown(wait=False, cancel_futures=True)
            for agent, future in zip(agents, futures):
                step: dict[str, Any] = {"role": agent["role"], "model_id": agent["model_id"]}
                if future in done:
                    try:
                        step["output"] = future.result()
                    except Exception as exc:  # noqa: BLE001 - report per-agent failures
                        step["output"] = ""
                        step["error"] = str(exc)
                else:
                    step["output"] = ""
                    step["error"] = "timed out"
                steps.append(step)
            if pending:
                return {"status": "timeout", "workflow": workflow, "steps": steps}
            synthesis = "\n\n".join(f"[{s['role']}]\n{s['output']}" for s in steps if s["output"])
            return {"status": "done", "workflow": workflow, "steps": steps, "synthesis": synthesis}

        if workflow == "review":
            draft = ""
            for _round in range(max_rounds):
                draft = self._run_agent(
                    agents[0]["model_id"],
                    agents[0]["role"],
                    [{"role": "user", "content": task + (f"\n\nPrevious draft:\n{draft}" if draft else "")}],
                    opts,
                )
                steps.append({"role": agents[0]["role"], "model_id": agents[0]["model_id"], "output": draft, "round": _round + 1})
                critique = self._run_agent(
                    agents[1]["model_id"],
                    agents[1]["role"],
                    [{"role": "user", "content": f"Task:\n{task}\n\nDraft to review:\n{draft}"}],
                    opts,
                )
                steps.append({"role": agents[1]["role"], "model_id": agents[1]["model_id"], "output": critique, "round": _round + 1})
            return {"status": "done", "workflow": workflow, "steps": steps, "synthesis": draft}

        # sequential (default): each agent sees the task plus prior contributions.
        context: list[dict[str, Any]] = [{"role": "user", "content": task}]
        for agent in agents:
            if time.monotonic() > deadline:
                return {"status": "timeout", "steps": steps}
            output = self._run_agent(agent["model_id"], agent["role"], context, opts)
            steps.append({"role": agent["role"], "model_id": agent["model_id"], "output": output})
            context.append({"role": "assistant", "content": f"[{agent['role']}] {output}"})
        return {
            "status": "done",
            "workflow": workflow,
            "steps": steps,
            "synthesis": steps[-1]["output"] if steps else "",
        }
