from __future__ import annotations

import os
from pathlib import Path


_ENV_VAR = "RAG_WORKSPACE_DIR"


def get_workspace_dir() -> Path:
    override = os.getenv(_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.cwd() / "exp_data").resolve()


def ensure_workspace_dir() -> Path:
    workspace_dir = get_workspace_dir()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def workspace_file(*parts: str) -> str:
    return str(ensure_workspace_dir().joinpath(*parts))
