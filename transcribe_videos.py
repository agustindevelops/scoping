"""Extract speaker-labeled transcripts for every video under WORKSPACE_PATH using WhisperX."""

from __future__ import annotations

import gc
import os
import shutil
import sys
import traceback
from pathlib import Path

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import get_writer

from workspace_paths import EDIT_DIR_NAME, TRANSCRIPT_DIR_NAME, transcript_dir

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".wmv",
}

FFMPEG_CANDIDATES = [
    Path(r"C:\Program Files\FFM\ffmpeg.exe"),
    Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
]

SKIP_DIR_NAMES = {EDIT_DIR_NAME, TRANSCRIPT_DIR_NAME, ".venv", "__pycache__"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is not set in .env")
    return value.strip().strip('"').strip("'")


def ensure_ffmpeg() -> Path:
    """Make sure ffmpeg is findable for WhisperX audio loading (esp. under VS Code debug)."""
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
            print(f"ffmpeg: {candidate}")
            return candidate

    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg or set FFMPEG_PATH in .env "
        r'(example: FFMPEG_PATH=C:\Program Files\FFM\ffmpeg.exe)'
    )


def resolve_device() -> str:
    configured = os.getenv("WHISPER_DEVICE", "").strip().lower()
    if configured in {"cuda", "cpu"}:
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(device: str) -> str:
    configured = os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
    if configured:
        return configured
    return "float16" if device == "cuda" else "int8"


def _is_under_skipped_dir(path: Path, workspace: Path) -> bool:
    try:
        relative = path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False
    return any(part in SKIP_DIR_NAMES for part in relative.parts)


def find_videos(workspace: Path, output_dir: Path) -> list[Path]:
    if workspace.is_file():
        if workspace.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Not a supported video file: {workspace}")
        return [workspace]

    if not workspace.is_dir():
        raise FileNotFoundError(f"WORKSPACE_PATH does not exist: {workspace}")

    output_resolved = output_dir.resolve()
    videos = sorted(
        path
        for path in workspace.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and output_resolved not in path.resolve().parents
        and not _is_under_skipped_dir(path, workspace)
    )
    return videos


def transcript_exists(output_dir: Path, stem: str, formats: list[str]) -> bool:
    return all((output_dir / f"{stem}.{fmt}").exists() for fmt in formats)


def write_outputs(
    result: dict,
    video: Path,
    output_dir: Path,
    formats: list[str],
) -> None:
    writer_options = {
        "max_line_width": None,
        "max_line_count": None,
        "highlight_words": False,
    }
    for fmt in formats:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(video), writer_options)


def transcribe_video(
    model,
    diarize_model: DiarizationPipeline,
    video: Path,
    *,
    device: str,
    batch_size: int,
    speaker_count: int,
    language: str | None,
) -> dict:
    print(f"Loading audio: {video.name}")
    audio = whisperx.load_audio(str(video))

    print(f"Transcribing: {video.name}")
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
    )

    language_code = result.get("language") or language
    if result.get("segments") and language_code:
        print(f"Aligning ({language_code}): {video.name}")
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        result["language"] = language_code
        del align_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Diarizing ({speaker_count} speakers): {video.name}")
    diarize_segments = diarize_model(
        audio,
        num_speakers=speaker_count,
        min_speakers=speaker_count,
        max_speakers=speaker_count,
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result


def resolve_workspace_path() -> Path:
    raw = os.getenv("WORKSPACE_PATH") or os.getenv("VIDEO_PATH")
    if raw is None or not str(raw).strip():
        raise ValueError("WORKSPACE_PATH is not set in .env")
    return Path(str(raw).strip().strip('"').strip("'")).expanduser().resolve()


def main() -> int:
    load_dotenv()

    try:
        ensure_ffmpeg()
        workspace = resolve_workspace_path()
        hf_token = require_env("HF_TOKEN")
        speaker_count = int(require_env("SPEAKER_COUNT"))
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if speaker_count < 1:
        print("ERROR: SPEAKER_COUNT must be >= 1", file=sys.stderr)
        return 1

    if workspace.is_file():
        output_dir = workspace.parent / TRANSCRIPT_DIR_NAME
    else:
        output_dir = transcript_dir(workspace)

    model_name = os.getenv("WHISPER_MODEL", "large-v2")
    batch_size = int(
        os.getenv("WHISPER_BATCH_SIZE", "8" if torch.cuda.is_available() else "4")
    )
    language = os.getenv("WHISPER_LANGUAGE") or None
    skip_existing = env_bool("WHISPER_SKIP_EXISTING", True)
    formats = [
        fmt.strip().lower()
        for fmt in os.getenv("WHISPER_OUTPUT_FORMATS", "txt,json,srt").split(",")
        if fmt.strip()
    ]

    device = resolve_device()
    compute_type = resolve_compute_type(device)

    videos = find_videos(workspace if workspace.is_dir() else workspace.parent, output_dir)
    if workspace.is_file():
        videos = [workspace]

    if not videos:
        print(f"No videos found under {workspace}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"WORKSPACE_PATH: {workspace}")
    print(f"Transcript dir: {output_dir}")
    print(f"Speakers:       {speaker_count}")
    print(f"Videos:         {len(videos)}")
    print(f"Skip existing:  {skip_existing}")
    print(f"Model:          {model_name} ({device}, {compute_type})")
    print(f"Formats:        {', '.join(formats)}")

    pending = [
        video
        for video in videos
        if not (skip_existing and transcript_exists(output_dir, video.stem, formats))
    ]
    if skip_existing and len(pending) < len(videos):
        print(f"Already transcribed: {len(videos) - len(pending)}")
    if not pending:
        print("Nothing to transcribe.")
        return 0

    print("Loading WhisperX model...")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )

    diarize_model_name = os.getenv(
        "DIARIZE_MODEL",
        "pyannote/speaker-diarization-community-1",
    )
    print(f"Loading diarization model ({diarize_model_name})...")
    try:
        diarize_model = DiarizationPipeline(
            model_name=diarize_model_name,
            token=hf_token,
            device=device,
        )
    except Exception as exc:
        message = str(exc)
        if "GatedRepoError" in type(exc).__name__ or "403" in message or "gated" in message.lower():
            print(
                "\nERROR: Hugging Face blocked the diarization model (gated repo).\n"
                "Do this while logged into the same account that owns HF_TOKEN:\n"
                f"  1. Open https://huggingface.co/{diarize_model_name}\n"
                "  2. Accept the user conditions / request access\n"
                "  3. Also accept any linked gated models shown on that page\n"
                "  4. Confirm HF_TOKEN is a valid read token from that account\n"
                "  5. Re-run this launch config\n",
                file=sys.stderr,
            )
        raise

    failed: list[str] = []
    for index, video in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {video.name}")

        try:
            result = transcribe_video(
                model,
                diarize_model,
                video,
                device=device,
                batch_size=batch_size,
                speaker_count=speaker_count,
                language=language,
            )
            write_outputs(result, video, output_dir, formats)
            print(f"Saved to {output_dir}")
        except Exception as exc:  # noqa: BLE001 - keep batching other videos
            failed.append(video.name)
            print(f"FAILED {video.name}: {exc}", file=sys.stderr)
            traceback.print_exc()

    del model
    del diarize_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if failed:
        print(f"\nCompleted with {len(failed)} failure(s): {', '.join(failed)}")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
