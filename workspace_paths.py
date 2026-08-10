"""Agnostic workspace path helpers for edit jobs and transcripts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

EDIT_DIR_NAME = "edit"
TRANSCRIPT_DIR_NAME = "transcript"


@dataclass(frozen=True)
class JobPaths:
    slug: str
    dir: Path
    video: Path
    document: Path
    config: Path


def slugify_title(title: str) -> str:
    """Lowercase kebab-case safe folder/file stem from a job title."""
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive slug from title: {title!r}")
    return slug


def job_paths(workspace: Path, title: str) -> JobPaths:
    """Resolve edit/{slug}/ paths for a job title under the workspace."""
    workspace = workspace.expanduser().resolve()
    slug = slugify_title(title)
    job_dir = workspace / EDIT_DIR_NAME / slug
    return JobPaths(
        slug=slug,
        dir=job_dir,
        video=job_dir / f"{slug}.mp4",
        document=job_dir / f"{slug}.md",
        config=job_dir / "config.json",
    )


def transcript_dir(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / TRANSCRIPT_DIR_NAME


def edit_dir(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / EDIT_DIR_NAME


def resolve_workspace_video(workspace: Path, video_path: str | Path) -> Path:
    path = Path(video_path).expanduser()
    if not path.is_absolute():
        path = workspace.expanduser().resolve() / path
    return path.resolve()
