#!/usr/bin/env python3
"""
wall2tex: watch a camera, read what is handwritten on a wall/whiteboard,
convert it to LaTeX with a vision model, and push it to an Overleaf project.
 
Pipeline
  camera -> wait until the scene is still (no hand/body in frame)
         -> if the writing changed since the last upload
         -> vision model transcribes the full surface to LaTeX
         -> wall_notes.tex is rewritten in a local clone of the Overleaf project
         -> git commit + push (Overleaf Git integration)
 
In your main.tex add:   \\input{wall_notes}
"""
 
import argparse
import base64
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
 
import cv2
import numpy as np
from anthropic import Anthropic
 
MODEL = os.environ.get("WALL2TEX_MODEL", "claude-sonnet-5-5")
NOTES_FILE = "wall_notes.tex"
 
PROMPT = r"""This image shows handwriting on a wall or whiteboard.
Transcribe everything written into LaTeX that compiles inside an existing
document body (no \documentclass, no \begin{document}).
 
Rules:
- Use $...$ for inline math and \begin{equation}/\begin{align*} for displayed equations,
  following the layout of the writing (one line of writing -> one line of LaTeX).
- Plain words stay as text. Lists stay as itemize/enumerate.
- Sketches or diagrams: add a comment line `% [diagram: short description]`.
- Anything you cannot read with confidence: `% [illegible]` or `% [uncertain: your best guess]`.
- Ignore the wall, shadows, marker trays, and people.
- Output only the LaTeX, no explanations and no code fences."""
 
 
# ---------------------------------------------------------------- camera ----
def open_camera(source: str) -> cv2.VideoCapture:
    # "0", "1" -> local webcam index; otherwise treat as URL (phone IP-camera app, RTSP, ...)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap
 
 
def small_gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (320, 180))
    return cv2.GaussianBlur(g, (5, 5), 0)
 
 
def diff_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(a, b)))
 
 
def enhance(frame: np.ndarray) -> np.ndarray:
    """Even out lighting so faint marker strokes survive JPEG compression."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
 
 
def crop_to_roi(frame: np.ndarray, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]
 
 
def pick_roi(cap: cv2.VideoCapture):
    print("Draw a box around the writing surface, then press ENTER (c to cancel).")
    ok, frame = cap.read()
    if not ok:
        return None
    roi = cv2.selectROI("Select writing surface", frame, showCrosshair=False)
    cv2.destroyWindow("Select writing surface")
    return roi if roi[2] > 0 and roi[3] > 0 else None
 
 
# ---------------------------------------------------------- transcription ---
def transcribe(client: Anthropic, frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", enhance(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # strip accidental fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
 
 
def looks_balanced(tex: str) -> bool:
    """Cheap sanity check so a broken transcription does not break the Overleaf build."""
    body = tex.replace(r"\$", "")
    return (tex.count("{") == tex.count("}")
            and body.count("$") % 2 == 0
            and tex.count(r"\begin{") == tex.count(r"\end{"))
 
 
# --------------------------------------------------------------- overleaf ---
def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
 
 
def prepare_repo(repo_dir: Path, project_id: str | None, token: str | None) -> bool:
    """Clone the Overleaf project through its Git bridge. Returns True if pushing is possible."""
    if not project_id or not token:
        repo_dir.mkdir(parents=True, exist_ok=True)
        print("No Overleaf credentials: writing locally only (see README for setup).")
        return False
    if not (repo_dir / ".git").exists():
        url = f"https://git:{token}@git.overleaf.com/{project_id}"
        run(["git", "clone", url, str(repo_dir)], cwd=".")
    run(["git", "pull", "--rebase"], cwd=repo_dir)
    return True
 
 
def publish(repo_dir: Path, tex: str, can_push: bool, history_dir: Path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{stamp}.tex").write_text(tex, encoding="utf-8")
 
    header = f"% Auto-generated by wall2tex at {stamp}. Overwritten on each update.\n"
    (repo_dir / NOTES_FILE).write_text(header + tex + "\n", encoding="utf-8")
 
    if not can_push:
        print(f"[{stamp}] saved {repo_dir / NOTES_FILE}")
        return
    run(["git", "add", NOTES_FILE], cwd=repo_dir)
    status = run(["git", "status", "--porcelain"], cwd=repo_dir).stdout.strip()
    if not status:
        return
    run(["git", "commit", "-m", f"wall2tex update {stamp}"], cwd=repo_dir)
    try:
        run(["git", "pull", "--rebase"], cwd=repo_dir)
        run(["git", "push"], cwd=repo_dir)
        print(f"[{stamp}] pushed to Overleaf")
    except subprocess.CalledProcessError as e:
        print("Git push failed:", e.stderr.strip())
 
 
# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="webcam index or stream URL")
    ap.add_argument("--repo-dir", default="overleaf_repo")
    ap.add_argument("--still-seconds", type=float, default=2.5,
                    help="how long the scene must be motionless before it is read")
    ap.add_argument("--motion-threshold", type=float, default=2.0,
                    help="mean pixel difference that counts as movement")
    ap.add_argument("--change-threshold", type=float, default=3.0,
                    help="difference from the last uploaded view that counts as new writing")
    ap.add_argument("--select-roi", action="store_true", help="crop to the writing surface")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
 
    client = Anthropic()  # reads ANTHROPIC_API_KEY
    repo_dir = Path(args.repo_dir)
    can_push = prepare_repo(repo_dir, os.environ.get("OVERLEAF_PROJECT_ID"),
                            os.environ.get("OVERLEAF_GIT_TOKEN"))
 
    cap = open_camera(args.camera)
    roi = pick_roi(cap) if args.select_roi else None
 
    prev = None            # previous frame (small gray)
    still_since = None     # when the scene last became motionless
    last_sent = None       # small gray of the last transcribed view
    busy_until = 0.0
 
    print("Watching. Press q in the preview window (or Ctrl+C) to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            frame = crop_to_roi(frame, roi)
            g = small_gray(frame)
            now = time.time()
            state = "moving"
 
            if prev is not None and diff_score(g, prev) < args.motion_threshold:
                still_since = still_since or now
                state = "still"
                ready = (now - still_since) >= args.still_seconds and now > busy_until
                changed = last_sent is None or diff_score(g, last_sent) > args.change_threshold
                if ready and changed:
                    state = "reading"
                    try:
                        tex = transcribe(client, frame)
                        if not tex:
                            print("Nothing legible found.")
                        elif not looks_balanced(tex):
                            print("Transcription has unbalanced braces/math; skipped. Will retry on next change.")
                        else:
                            publish(repo_dir, tex, can_push, repo_dir.parent / "wall2tex_history")
                            last_sent = g
                    except Exception as e:  # network, API, git
                        print("Error:", e)
                    busy_until = time.time() + 5  # cool-down between requests
            else:
                still_since = None
            prev = g
 
            if not args.no_preview:
                view = frame.copy()
                cv2.putText(view, state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("wall2tex", cv2.resize(view, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
