# scoping

Content-agnostic video transcription and edit tooling. Point `WORKSPACE_PATH` at a project folder of source videos.

## Workspace layout

```text
workspace/
  *.mp4
  transcript/
  edit/
    notes.csv                 # optional
    {slug}/
      config.json             # array of edit jobs (validated by schemas/edit_config.schema.json)
      {slug}.mp4
      {slug}.md
```

## Commands

```bash
# Transcribe new videos only (skips existing transcript outputs)
.venv/Scripts/python.exe transcribe_videos.py

# Render all edit/**/config.json jobs (skips if {slug}.mp4 already exists)
.venv/Scripts/python.exe edit_videos.py
```

Job output paths come from `workspace_paths.job_paths(workspace, title)` — never hard-code project names in the tooling repo.

Edit configs use a flat `videos` array (each item is one clip with its own `video_path`), so you can interleave sources freely. Job-level `labels` are reserved for on-screen captions (pycaps).
