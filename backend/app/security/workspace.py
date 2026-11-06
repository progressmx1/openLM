"""Workspace boundary enforcement.

File tools may only touch paths inside user-approved workspace directories unless
the user explicitly approves an out-of-workspace action. Paths are canonicalized
(expanded, ``..`` resolved, symlinks/junctions followed) before containment is
checked, which blocks path traversal and symlink/junction escapes.
"""
from __future__ import annotations

import os
from pathlib import Path


class WorkspaceViolation(Exception):
    """Raised when a path escapes an approved workspace boundary."""


def canonicalize(path: str | Path) -> Path:
    """Return an absolute, normalized path with symlinks resolved (best effort)."""
    return Path(os.path.abspath(os.path.expanduser(str(path)))).resolve(strict=False)


def _is_case_insensitive() -> bool:
    return os.name == "nt"


def _key(path: Path) -> str:
    return os.path.normcase(str(path)) if _is_case_insensitive() else str(path)


class WorkspacePolicy:
    def __init__(self, approved_workspaces: list[str] | None = None) -> None:
        roots: list[Path] = []
        for raw in approved_workspaces or []:
            if raw and raw.strip():
                roots.append(canonicalize(raw.strip()))
        self._roots = roots

    @property
    def roots(self) -> list[str]:
        return [str(r) for r in self._roots]

    def has_workspaces(self) -> bool:
        return bool(self._roots)

    def is_within(self, path: str | Path) -> bool:
        """True if ``path`` is equal to or nested inside any approved workspace."""
        target = canonicalize(path)
        target_key = _key(target)
        for root in self._roots:
            root_key = _key(root)
            if target_key == root_key:
                return True
            if target_key.startswith(root_key + os.sep):
                return True
        return False

    def resolve_in_workspace(self, base: str | Path, *parts: str) -> Path:
        """Join ``parts`` onto an approved ``base`` and verify containment.

        Raises :class:`WorkspaceViolation` if the result escapes ``base``.
        """
        base_c = canonicalize(base)
        candidate = base_c.joinpath(*parts)
        candidate = canonicalize(candidate)
        base_key = _key(base_c)
        candidate_key = _key(candidate)
        if candidate_key != base_key and not candidate_key.startswith(base_key + os.sep):
            raise WorkspaceViolation(
                f"Path '{candidate}' escapes the approved workspace '{base_c}'."
            )
        return candidate

    def assert_within(self, path: str | Path) -> Path:
        """Canonicalize ``path`` and raise if it is outside all approved workspaces."""
        target = canonicalize(path)
        if not self.is_within(target):
            raise WorkspaceViolation(
                f"Path '{target}' is outside all approved workspaces."
            )
        return target

    def choose_base(self, path: str | Path) -> Path:
        """Return the approved workspace root that contains ``path`` (or raise)."""
        target = canonicalize(path)
        target_key = _key(target)
        for root in self._roots:
            root_key = _key(root)
            if target_key == root_key or target_key.startswith(root_key + os.sep):
                return root
        raise WorkspaceViolation(
            f"Path '{target}' is outside all approved workspaces."
        )
