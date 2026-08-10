"""Cut and stitch workspace edit jobs from validated configs; write edit documents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from edit_schema import load_edit_config
from workspace_paths import (
    edit_dir,
    job_paths,
    resolve_workspace_video,
)

TIME_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?$"
)

LABEL_FADE_SECONDS = 0.35


def log(message: str = "") -> None:
    print(message, flush=True)


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


@dataclass(frozen=True)
class Clip:
    video_path: Path
    start: float
    end: float
    title: str
    description: str
    transcript_cue: str
    index: int
    edited_start: float
    edited_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Label:
    text: str
    start: float
    end: float


def parse_time(value: str | int | float, *, field: str) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"{field} must be >= 0, got {value}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is empty")

    match = TIME_RE.match(text)
    if not match:
        raise ValueError(
            f"{field} must be seconds or H:MM:SS / M:SS (optional ms), got {value!r}"
        )

    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    ms_raw = match.group("ms") or "0"
    millis = int(ms_raw.ljust(3, "0")[:3])
    total = hours * 3600 + minutes * 60 + seconds + millis / 1000
    if total < 0:
        raise ValueError(f"{field} must be >= 0, got {value}")
    return total


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:d}:{secs:02d}.{millis:03d}"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def resolve_workspace() -> Path:
    raw = os.getenv("WORKSPACE_PATH") or os.getenv("VIDEO_PATH")
    if raw is None or not str(raw).strip():
        raise ValueError("WORKSPACE_PATH is not set in .env")
    path = Path(str(raw).strip().strip('"').strip("'")).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"WORKSPACE_PATH does not exist: {path}")
    if path.is_file():
        return path.parent
    return path


def discover_config_files(workspace: Path) -> list[Path]:
    root = edit_dir(workspace)
    if not root.is_dir():
        return []
    return sorted(root.rglob("config.json"))


def parse_job_clips(job: dict[str, Any], workspace: Path) -> list[Clip]:
    clips: list[Clip] = []
    cursor = 0.0
    clip_index = 0

    for item_i, item in enumerate(job["videos"], start=1):
        if item.get("enabled") is False:
            continue

        video_path = resolve_workspace_video(workspace, str(item["video_path"]))
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start = parse_time(item["start"], field=f"videos[{item_i - 1}].start")
        end = parse_time(item["end"], field=f"videos[{item_i - 1}].end")
        if end <= start:
            raise ValueError(
                f"videos[{item_i - 1}]: end ({item['end']}) must be after start ({item['start']})"
            )

        clip_index += 1
        clips.append(
            Clip(
                video_path=video_path,
                start=start,
                end=end,
                title=str(item.get("title") or f"Clip {clip_index}"),
                description=str(item.get("description") or "").strip(),
                transcript_cue=str(item.get("transcript_cue") or "").strip(),
                index=clip_index,
                edited_start=cursor,
                edited_end=cursor + (end - start),
            )
        )
        cursor += end - start

    if not clips:
        raise ValueError("No enabled clips found in job")
    return clips


def parse_job_labels(job: dict[str, Any]) -> list[Label]:
    labels: list[Label] = []
    for i, item in enumerate(job.get("labels") or []):
        start = parse_time(item["start"], field=f"labels[{i}].start")
        end = parse_time(item["end"], field=f"labels[{i}].end")
        if end <= start:
            raise ValueError(
                f"labels[{i}]: end ({item['end']}) must be after start ({item['start']})"
            )
        labels.append(Label(text=str(item["text"]), start=start, end=end))
    return labels


def write_edit_document(
    job: dict[str, Any],
    clips: list[Clip],
    labels: list[Label],
    output_video: Path,
    document_path: Path,
) -> None:
    title = str(job["title"])
    overview = str(job.get("description") or "").strip()
    total = sum(clip.duration for clip in clips)

    lines = [
        f"# {title}",
        "",
        f"**Output:** `{output_video}`",
        f"**Clips:** {len(clips)}",
        f"**Approx. duration:** {format_duration(total)}",
        "",
    ]
    if overview:
        lines.extend(["## Overview", "", overview, ""])

    if labels:
        lines.extend(["## Labels", ""])
        for label in labels:
            lines.append(
                f"- `{format_time(label.start)}` → `{format_time(label.end)}`: {label.text}"
            )
        lines.append("")

    lines.extend(["## Edit order", ""])

    for clip in clips:
        lines.append(f"### {clip.index}. {clip.title}")
        lines.append("")
        lines.append(
            f"- Edited timeline: `{format_time(clip.edited_start)}` → `{format_time(clip.edited_end)}`"
        )
        lines.append(
            f"- Source: `{clip.video_path.name}` "
            f"`{format_time(clip.start)}` → `{format_time(clip.end)}` "
            f"({format_duration(clip.duration)})"
        )
        if clip.transcript_cue:
            lines.append(f"- Transcript cue: {clip.transcript_cue}")
        if clip.description:
            lines.append(f"- Notes: {clip.description}")
        lines.append("")

    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _make_label_clip(text: str, duration: float, size: tuple[int, int]):
    from moviepy import TextClip
    from moviepy.video.fx import CrossFadeIn, CrossFadeOut

    width, height = size
    fontsize = max(28, min(64, width // 28))
    fade = min(LABEL_FADE_SECONDS, max(0.05, duration / 3))

    try:
        txt = TextClip(
            text=text,
            font_size=fontsize,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(int(width * 0.85), None),
            text_align="center",
            duration=duration,
        )
    except TypeError:
        # Older / alternate TextClip signatures
        txt = TextClip(
            text,
            fontsize=fontsize,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(int(width * 0.85), None),
            align="center",
            duration=duration,
        )

    txt = txt.with_position(("center", int(height * 0.12)))
    return txt.with_effects([CrossFadeIn(fade), CrossFadeOut(fade)])


def render_job(
    clips: list[Clip],
    labels: list[Label],
    output_video: Path,
) -> None:
    from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips

    total_clips = len(clips)
    total_duration = sum(clip.duration for clip in clips)
    started = time.perf_counter()
    pieces = []
    sources: list[Any] = []

    log(
        f"\n[1/4] Cutting {total_clips} clips "
        f"(~{format_duration(total_duration)} total source length)..."
    )
    try:
        cut_started = time.perf_counter()
        for index, clip in enumerate(clips, start=1):
            clip_started = time.perf_counter()
            remaining = total_clips - index + 1
            log(
                f"  [{index}/{total_clips}] {clip.title} "
                f"({format_duration(clip.duration)}) "
                f"from `{clip.video_path.name}` "
                f"{format_time(clip.start)} → {format_time(clip.end)} "
                f"| remaining clips: {remaining}"
            )
            source = VideoFileClip(str(clip.video_path))
            sources.append(source)
            piece = source.subclipped(clip.start, clip.end)
            pieces.append(piece)
            log(
                f"       done in {format_elapsed(time.perf_counter() - clip_started)} "
                f"(elapsed {format_elapsed(time.perf_counter() - cut_started)})"
            )

        log(
            f"[1/4] Cuts finished in {format_elapsed(time.perf_counter() - cut_started)}"
        )

        log("[2/4] Concatenating timeline...")
        concat_started = time.perf_counter()
        timeline = concatenate_videoclips(pieces, method="compose")
        log(
            f"[2/4] Concatenate done in "
            f"{format_elapsed(time.perf_counter() - concat_started)} "
            f"(timeline ~{format_duration(float(timeline.duration or total_duration))})"
        )

        log(f"[3/4] Applying {len(labels)} label overlay(s)...")
        label_started = time.perf_counter()
        overlays = []
        for label_i, label in enumerate(labels, start=1):
            duration = label.end - label.start
            if duration <= 0:
                continue
            log(
                f"  label [{label_i}/{len(labels)}] "
                f"{format_time(label.start)} → {format_time(label.end)}: {label.text}"
            )
            overlay = _make_label_clip(label.text, duration, timeline.size)
            overlays.append(overlay.with_start(label.start).with_end(label.end))
        final = CompositeVideoClip([timeline, *overlays]) if overlays else timeline
        log(
            f"[3/4] Labels done in {format_elapsed(time.perf_counter() - label_started)}"
        )

        output_video.parent.mkdir(parents=True, exist_ok=True)
        log(
            f"[4/4] Encoding `{output_video.name}` "
            f"(~{format_duration(float(final.duration or total_duration))})..."
        )
        encode_started = time.perf_counter()
        final.write_videofile(
            str(output_video),
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
            logger="bar",
        )
        log(
            f"[4/4] Encode finished in "
            f"{format_elapsed(time.perf_counter() - encode_started)}"
        )
        log(f"Render complete in {format_elapsed(time.perf_counter() - started)}")
    finally:
        for piece in pieces:
            try:
                piece.close()
            except Exception:  # noqa: BLE001
                pass
        for source in sources:
            try:
                source.close()
            except Exception:  # noqa: BLE001
                pass


def process_job(job: dict[str, Any], workspace: Path, config_path: Path) -> str:
    """Render one job into the folder that contains its config.json."""
    title = str(job["title"])
    job_dir = config_path.parent
    paths = job_paths(job_dir, title)

    if paths.video.is_file():
        log(f"Skipping (output exists): {paths.video}")
        return "skipped"

    clips = parse_job_clips(job, workspace)
    labels = parse_job_labels(job)
    total_duration = sum(clip.duration for clip in clips)

    log(f"Job:      {title}")
    log(f"Folder:   {paths.dir}")
    log(f"Output:   {paths.video}")
    log(f"Document: {paths.document}")
    log(f"Clips:    {len(clips)}")
    log(f"Labels:   {len(labels)}")
    log(f"Length:   ~{format_duration(total_duration)}")

    job_started = time.perf_counter()
    render_job(clips, labels, paths.video)
    log("[doc] Writing edit document...")
    write_edit_document(job, clips, labels, paths.video, paths.document)
    log(f"Wrote {paths.video}")
    log(f"Wrote {paths.document}")
    log(f"Job finished in {format_elapsed(time.perf_counter() - job_started)}")
    return "rendered"


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Validate workspace edit configs, stitch clips with moviepy, "
            "and write per-job outputs under edit/{slug}/."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.getenv("EDIT_CONFIG", ""),
        help="Optional path to a config.json (default: discover WORKSPACE_PATH/edit/**/config.json)",
    )
    args = parser.parse_args()

    try:
        workspace = resolve_workspace()
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.config:
        config_files = [Path(args.config).expanduser().resolve()]
        missing = [path for path in config_files if not path.is_file()]
        if missing:
            print(f"ERROR: config not found: {missing[0]}", file=sys.stderr)
            return 1
    else:
        config_files = discover_config_files(workspace)
        if not config_files:
            log(f"No edit configs found under {edit_dir(workspace)}")
            return 0

    run_started = time.perf_counter()
    log(f"WORKSPACE_PATH: {workspace}")
    log(f"Configs:        {len(config_files)}")

    rendered = 0
    skipped = 0
    failed = 0
    total_jobs = 0

    for config_path in config_files:
        log(f"\n=== {config_path} ===")
        try:
            jobs = load_edit_config(config_path)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failed += 1
            continue

        total_jobs += len(jobs)
        for job_i, job in enumerate(jobs, start=1):
            log(f"\n--- Job {job_i}/{len(jobs)} ---")
            try:
                status = process_job(job, workspace, config_path)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"ERROR ({job.get('title', '?')}): {exc}", file=sys.stderr)
                continue
            if status == "skipped":
                skipped += 1
            else:
                rendered += 1

    log(
        f"\nDone in {format_elapsed(time.perf_counter() - run_started)}. "
        f"jobs={total_jobs} rendered={rendered} skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
