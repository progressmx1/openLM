"""Tests for the multi-model coordinator."""
from __future__ import annotations

from app.agents.multi_agent import MultiAgentCoordinator
from app.models.registry import ModelRegistry

from .fake_adapter import FakeAdapter


class ScriptedAdapter(FakeAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.chat_calls: list[str] = []

    def chat(self, model_id, messages, options):
        self.chat_calls.append(model_id)
        return {"content": self.responses.pop(0) if self.responses else "done", "usage": {}}


def _setup(repository, settings_store, responses):
    adapter = ScriptedAdapter(responses)
    adapter.discovered = [
        {"name": "a", "path": "/m/a.gguf", "format": "gguf"},
        {"name": "b", "path": "/m/b.gguf", "format": "gguf"},
    ]
    settings = settings_store.load()
    settings.allow_multi_model = True
    settings_store.save(settings)
    model_registry = ModelRegistry(repository, settings_store, adapter=adapter)
    model_registry.discover(["x"])
    ids = [m["id"] for m in model_registry.list_models()]
    for model_id in ids:
        model_registry.load(model_id)
    coordinator = MultiAgentCoordinator(model_registry, settings_store)
    return coordinator, ids


def test_sequential_workflow(repository, settings_store) -> None:
    coordinator, ids = _setup(repository, settings_store, ["plan", "final"])
    result = coordinator.run(
        "write a poem",
        [{"model_id": ids[0], "role": "Planner"}, {"model_id": ids[1], "role": "Writer"}],
        "sequential",
    )
    assert result["status"] == "done"
    assert result["synthesis"] == "final"
    assert len(result["steps"]) == 2


def test_review_workflow(repository, settings_store) -> None:
    coordinator, ids = _setup(repository, settings_store, ["draft", "critique"])
    result = coordinator.run(
        "task",
        [{"model_id": ids[0], "role": "Writer"}, {"model_id": ids[1], "role": "Reviewer"}],
        "review",
        max_rounds=1,
    )
    assert result["status"] == "done"
    assert result["synthesis"] == "draft"
    assert len(result["steps"]) == 2


def test_parallel_workflow(repository, settings_store) -> None:
    coordinator, ids = _setup(repository, settings_store, ["alpha", "beta"])
    result = coordinator.run(
        "analyze",
        [{"model_id": ids[0], "role": "A"}, {"model_id": ids[1], "role": "B"}],
        "parallel",
    )
    assert result["status"] == "done"
    assert "[A]" in result["synthesis"]
    assert "[B]" in result["synthesis"]


def test_unloaded_model_errors(repository, settings_store) -> None:
    adapter = ScriptedAdapter(["x"])
    adapter.discovered = [
        {"name": "a", "path": "/m/a.gguf", "format": "gguf"},
        {"name": "b", "path": "/m/b.gguf", "format": "gguf"},
    ]
    model_registry = ModelRegistry(repository, settings_store, adapter=adapter)
    model_registry.discover(["x"])
    ids = [m["id"] for m in model_registry.list_models()]
    # Do not load the models.
    coordinator = MultiAgentCoordinator(model_registry, settings_store)
    result = coordinator.run(
        "task",
        [{"model_id": ids[0], "role": "A"}, {"model_id": ids[1], "role": "B"}],
    )
    assert result["status"] == "error"
    assert "not loaded" in result["error"]


def test_requires_two_agents(repository, settings_store) -> None:
    coordinator, ids = _setup(repository, settings_store, ["x"])
    result = coordinator.run("task", [{"model_id": ids[0], "role": "A"}])
    assert result["status"] == "error"
