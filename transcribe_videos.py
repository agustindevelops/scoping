"""Extract speaker-labeled transcripts for every video under VIDEO_PATH using WhisperX."""

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


def find_videos(video_path: Path, output_dir: Path) -> list[Path]:
    if video_path.is_file():
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Not a supported video file: {video_path}")
        return [video_path]

    if not video_path.is_dir():
        raise FileNotFoundError(f"VIDEO_PATH does not exist: {video_path}")

    output_resolved = output_dir.resolve()
    videos = sorted(
        path
        for path in video_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and output_resolved not in path.resolve().parents
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


def main() -> int:
    load_dotenv()

    try:
        ensure_ffmpeg()
        video_path_value = require_env("VIDEO_PATH")
        hf_token = require_env("HF_TOKEN")
        speaker_count = int(require_env("SPEAKER_COUNT"))
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if speaker_count < 1:
        print("ERROR: SPEAKER_COUNT must be >= 1", file=sys.stderr)
        return 1

    video_path = Path(video_path_value).expanduser().resolve()
    # Always write under VIDEO_PATH/transcript
    output_dir = video_path / "transcript"
    if video_path.is_file():
        output_dir = video_path.parent / "transcript"

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

    videos = find_videos(video_path, output_dir)
    if not videos:
        print(f"No videos found under {video_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"VIDEO_PATH:     {video_path}")
    print(f"Transcript dir: {output_dir}")
    print(f"Speakers:       {speaker_count}")
    print(f"Videos:         {len(videos)}")
    print(f"Model:          {model_name} ({device}, {compute_type})")
    print(f"Formats:        {', '.join(formats)}")

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
    for index, video in enumerate(videos, start=1):
        stem = video.stem
        print(f"\n[{index}/{len(videos)}] {video.name}")

        if skip_existing and transcript_exists(output_dir, stem, formats):
            print("Skipping (already transcribed)")
            continue

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
