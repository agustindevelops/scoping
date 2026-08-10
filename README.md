# scoping

Content-agnostic video transcription and edit tooling. Point `WORKSPACE_PATH` at a project folder of source videos.

## Workspace layout

```text
workspace/
  *.mp4
  transcript/
  edit/
    notes.csv                 # optional
    the-scope/                # you name this folder
      config.json             # array of edit jobs (validated by schemas/edit_config.schema.json)
      {title-slug}.mp4        # filename from config title
      {title-slug}.md
```

## Commands

```bash
# Transcribe new videos only (skips existing transcript outputs)
.venv/Scripts/python.exe transcribe_videos.py

# Render all edit/**/config.json jobs (skips if output mp4 already exists)
.venv/Scripts/python.exe edit_videos.py
```

The edit folder name is yours (`edit/the-scope/`). Output basenames come from `workspace_paths.job_paths(job_dir, title)` using the job `title` in `config.json` — the tooling never renames your folder.

Edit configs use a flat `videos` array (each item is one clip with its own `video_path`), so you can interleave sources freely. Job-level `labels` are reserved for on-screen captions (pycaps).
