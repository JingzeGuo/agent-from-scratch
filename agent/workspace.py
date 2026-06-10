from pathlib import Path


def resolve_workspace_path(workspace_root: Path, path: str) -> Path:
    """Resolve a path and ensure it remains inside the workspace."""
    root = workspace_root.expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path is outside the workspace: {path}")

    return resolved
