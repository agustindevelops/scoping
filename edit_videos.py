"""Cut and concatenate video clips from an ordered edit config, and write an edit document."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

FFMPEG_CANDIDATES = [
    Path(r"C:\Program Files\FFM\ffmpeg.exe"),
    Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
]

TIME_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?$"
)


@dataclass(frozen=True)
class Clip:
    video_path: Path
    video_label: str
    start: float
    end: float
    title: str
    description: str
    target_length: str
    transcript_cue: str
    index: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def ensure_ffmpeg() -> Path:
    configured = os.getenv("FFMPEG_PATH", "").strip().strip('"').strip("'")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    which = shutil.which("ffmpeg")
    if which:
        candidates.append(Path(which))

    candidates.extend(FFMPEG_CANDIDATES)

    for candidate in candidates:
        if candidate.is_file():
            ffmpeg_dir = str(candidate.parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if ffmpeg_dir not in path_parts:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return candidate

    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg or set FFMPEG_PATH in .env "
        r'(example: FFMPEG_PATH=C:\Program Files\FFM\ffmpeg.exe)'
    )


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


def resolve_video_path(raw: str, video_root: Path | None) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() and video_root is not None:
        path = video_root / path
    return path.resolve()


def load_config(path: Path) -> tuple[dict, list[Clip]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")

    videos = data.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Config must include a non-empty 'videos' array")

    video_root_raw = data.get("video_root") or os.getenv("VIDEO_PATH", "").strip()
    video_root = Path(video_root_raw).expanduser().resolve() if video_root_raw else None
    if video_root is not None and video_root.is_file():
        video_root = video_root.parent

    clips: list[Clip] = []
    clip_index = 0
    for video_i, video in enumerate(videos, start=1):
        if not isinstance(video, dict):
            raise ValueError(f"videos[{video_i - 1}] must be an object")

        raw_path = video.get("video_path")
        if not raw_path:
            raise ValueError(f"videos[{video_i - 1}].video_path is required")

        video_path = resolve_video_path(str(raw_path), video_root)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        label = str(video.get("label") or video_path.stem)
        timestamps = video.get("timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError(
                f"videos[{video_i - 1}] must include a non-empty 'timestamps' array"
            )

        for ts_i, item in enumerate(timestamps, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"videos[{video_i - 1}].timestamps[{ts_i - 1}] must be an object"
                )
            if item.get("enabled") is False:
                continue

            start = parse_time(item["start"], field=f"timestamps[{ts_i - 1}].start")
            end = parse_time(item["end"], field=f"timestamps[{ts_i - 1}].end")
            if end <= start:
                raise ValueError(
                    f"{label} clip {ts_i}: end ({item['end']}) must be after start ({item['start']})"
                )

            clip_index += 1
            clips.append(
                Clip(
                    video_path=video_path,
                    video_label=label,
                    start=start,
                    end=end,
                    title=str(item.get("title") or f"Clip {clip_index}"),
                    description=str(item.get("description") or "").strip(),
                    target_length=str(item.get("target_length") or "").strip(),
                    transcript_cue=str(item.get("transcript_cue") or "").strip(),
                    index=clip_index,
                )
            )

    if not clips:
        raise ValueError("No enabled clips found in config")

    return data, clips


def run_ffmpeg(ffmpeg: Path, args: list[str]) -> None:
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail}")


def cut_clip(ffmpeg: Path, clip: Clip, output: Path) -> None:
    # Input seeking first for speed; re-encode for accurate cuts across sources.
    run_ffmpeg(
        ffmpeg,
        [
            "-ss",
            f"{clip.start:.3f}",
            "-to",
            f"{clip.end:.3f}",
            "-i",
            str(clip.video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ],
    )


def concat_clips(ffmpeg: Path, parts: list[Path], output: Path) -> None:
    list_file = output.with_suffix(output.suffix + ".concat.txt")
    lines = []
    for part in parts:
        escaped = part.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run_ffmpeg(
            ffmpeg,
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
        )
    finally:
        list_file.unlink(missing_ok=True)


def write_edit_document(
    config: dict,
    clips: list[Clip],
    output_video: Path,
    document_path: Path,
) -> None:
    title = str(config.get("title") or "Video edit document")
    overview = str(config.get("description") or "").strip()
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

    lines.extend(["## Edit order", ""])

    current_label = None
    for clip in clips:
        if clip.video_label != current_label:
            current_label = clip.video_label
            lines.extend([f"### {current_label}", ""])
            # Carry video-level description from config when present
            for video in config.get("videos", []):
                label = str(video.get("label") or Path(str(video.get("video_path", ""))).stem)
                if label == current_label:
                    video_desc = str(video.get("description") or "").strip()
                    if video_desc:
                        lines.extend([video_desc, ""])
                    break

        lines.append(f"#### {clip.index}. {clip.title}")
        lines.append("")
        lines.append(
            f"- Source: `{clip.video_path.name}` "
            f"`{format_time(clip.start)}` → `{format_time(clip.end)}` "
            f"({format_duration(clip.duration)})"
        )
        if clip.target_length:
            lines.append(f"- Target edited length: {clip.target_length}")
        if clip.transcript_cue:
            lines.append(f"- Transcript cue: {clip.transcript_cue}")
        if clip.description:
            lines.append(f"- Notes: {clip.description}")
        lines.append("")

    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def resolve_output_paths(config: dict, config_path: Path) -> tuple[Path, Path]:
    base = config_path.parent
    output_raw = config.get("output") or config.get("output_path") or "edited/output.mp4"
    document_raw = (
        config.get("document")
        or config.get("document_path")
        or Path(str(output_raw)).with_suffix(".md")
    )

    output_path = Path(str(output_raw)).expanduser()
    document_path = Path(str(document_raw)).expanduser()
    if not output_path.is_absolute():
        output_path = (base / output_path).resolve()
    if not document_path.is_absolute():
        document_path = (base / document_path).resolve()
    return output_path, document_path


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Cut/concat clips from an ordered edit config and write an edit document."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.getenv("EDIT_CONFIG", ""),
        help="Path to edit config JSON (or set EDIT_CONFIG in .env)",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep intermediate cut files next to the output video",
    )
    args = parser.parse_args()

    if not args.config:
        print(
            "ERROR: pass a config path or set EDIT_CONFIG in .env",
            file=sys.stderr,
        )
        return 1

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        ffmpeg = ensure_ffmpeg()
        config, clips = load_config(config_path)
        output_path, document_path = resolve_output_paths(config, config_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"ffmpeg:   {ffmpeg}")
    print(f"config:   {config_path}")
    print(f"output:   {output_path}")
    print(f"document: {document_path}")
    print(f"clips:    {len(clips)}")

    parts_dir: Path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_parts:
        parts_dir = output_path.parent / f"{output_path.stem}_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="edit_videos_")
        parts_dir = Path(temp_dir.name)

    part_files: list[Path] = []
    try:
        for clip in clips:
            part = parts_dir / f"{clip.index:03d}.mp4"
            print(
                f"[{clip.index}/{len(clips)}] {clip.title} "
                f"({format_time(clip.start)} → {format_time(clip.end)})"
            )
            cut_clip(ffmpeg, clip, part)
            part_files.append(part)

        print("Concatenating...")
        concat_clips(ffmpeg, part_files, output_path)
        write_edit_document(config, clips, output_path, document_path)
        print(f"Wrote {output_path}")
        print(f"Wrote {document_path}")
        print("Done.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
