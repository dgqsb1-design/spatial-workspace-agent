# wall2tex

Camera -> handwriting -> LaTeX -> Overleaf.

## Install

```bash
pip install opencv-python anthropic numpy
export ANTHROPIC_API_KEY=sk-ant-...
```

## Connect Overleaf

Overleaf has no public write API. The reliable route is its Git integration
(requires a paid Overleaf plan, or one provided by your institution).

1. Overleaf -> Account Settings -> Git Integration -> create a token.
2. Project ID is the last part of the project URL: `overleaf.com/project/<ID>`.
3. Set both:

```bash
export OVERLEAF_PROJECT_ID=<ID>
export OVERLEAF_GIT_TOKEN=<token>
```

4. In your `main.tex`, add `\input{wall_notes}` where the wall content should appear.

Without Git access, run without those variables: the script writes
`overleaf_repo/wall_notes.tex` locally, and you upload that one file to the
project by hand (or sync it through Overleaf's Dropbox integration, which is
also a paid feature).

## Run

```bash
python wall2tex.py --camera 0 --select-roi
# phone as camera (IP Webcam / DroidCam style app):
python wall2tex.py --camera http://192.168.1.20:8080/video
```

`--select-roi` lets you draw a box around the writing area once at startup.

## How it behaves

- Waits until the scene is motionless for `--still-seconds` (so your hand or body is not in view).
- Compares against the last uploaded view; only reads again if the writing changed.
- Each read transcribes the whole surface and overwrites `wall_notes.tex`, so
  erasing and rewriting on the wall is reflected correctly. Every version is
  also kept in `wall2tex_history/`, and each push is a git commit in Overleaf's history.
- Skips a transcription whose braces, `$` or `\begin/\end` do not balance, so a
  bad read cannot break your Overleaf compile.

## Tuning

- Writing triggers too often or never: adjust `--motion-threshold` and `--change-threshold`.
- Poor reads: use dark markers, even lighting, and fill the frame with the writing (`--select-roi`).
- Change the model with `WALL2TEX_MODEL`.
