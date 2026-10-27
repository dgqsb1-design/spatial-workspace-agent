"""idea-capture: scan handwriting anywhere in a room on demand and save it as LaTeX + Word.

    python -m idea_capture                       # live camera, SPACE to scan
    python -m idea_capture --image photo.jpg     # no camera: scan a saved photo
    python -m idea_capture --camera http://192.168.1.20:8080/video   # phone as camera

SPACE in a preview window = scan, q = quit.
"""

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .engine import Engine
from .output import append_docx, append_tex


# ---------------------------------------------------------------- imaging ---
def open_camera(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def sharpness(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def grab_sharpest(cap: cv2.VideoCapture, n: int = 8):
    """Read several frames and keep the least blurred one (also flushes stale buffers)."""
    frames = [f for ok, f in (cap.read() for _ in range(n)) if ok]
    return max(frames, key=sharpness) if frames else None


def enhance(frame: np.ndarray) -> np.ndarray:
    """Even out lighting and glare so faint marker strokes stay readable."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def shrink(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame


# ------------------------------------------------------------------- scan ---
def process(engine, items, out_dir: Path, max_side: int):
    """items: list of (label, frame). Reads each, saves those that contain writing."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    saved = 0
    for label, frame in items:
        frame = shrink(enhance(frame), max_side)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            print(f"{label}: could not encode image")
            continue
        print(f"{label}: reading ... (a few seconds on a GPU, a minute or two on a CPU)")
        try:
            result = engine.read(buf.tobytes())
        except Exception as e:
            print(f"{label}: failed ({e})")
            continue
        blocks = result["blocks"]
        if not blocks:
            print(f"{label}: no writing found")
            continue
        image_path = out_dir / "images" / f"{stamp}_{label.replace(' ', '')}.jpg"
        cv2.imwrite(str(image_path), frame)
        append_tex(out_dir / "ideas.tex", stamp, label, blocks)
        append_docx(out_dir / "ideas.docx", stamp, label, blocks, image_path)
        print(f"{label}: saved {len(blocks)} blocks {[b['type'] for b in blocks]}")
        saved += 1
    if saved:
        print(f"-> {out_dir / 'ideas.docx'} and {out_dir / 'ideas.tex'}")


def voice_listener(trigger: threading.Event):
    """Offline keyword spotting for 'scan' (experimental).
    Needs: pip install SpeechRecognition pyaudio pocketsphinx"""
    import speech_recognition as sr
    rec, mic = sr.Recognizer(), sr.Microphone()
    with mic as src:
        rec.adjust_for_ambient_noise(src)
    print("Voice trigger on: say 'scan'.")
    while True:
        with mic as src:
            audio = rec.listen(src, phrase_time_limit=3)
        try:
            heard = rec.recognize_sphinx(audio, keyword_entries=[("scan", 0.8)])
        except Exception:
            continue
        if "scan" in heard.lower():
            trigger.set()


def main():
    ap = argparse.ArgumentParser(prog="idea_capture", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", action="append", help="webcam index or stream URL; repeat for several")
    ap.add_argument("--image", action="append", help="scan a saved photo instead of using a camera")
    ap.add_argument("--model", help="Hugging Face model id (default Qwen/Qwen3-VL-4B-Instruct)")
    ap.add_argument("--out", default="notes")
    ap.add_argument("--max-side", type=int, default=1280, help="longest image side given to the model")
    ap.add_argument("--voice", action="store_true")
    args = ap.parse_args()

    engine = Engine(args.model)
    out_dir = Path(args.out)

    if args.image:
        items = []
        for p in args.image:
            img = cv2.imread(p)
            if img is None:
                sys.exit(f"Cannot read image: {p}")
            items.append((Path(p).stem, img))
        process(engine, items, out_dir, args.max_side)
        return

    caps = [open_camera(s) for s in (args.camera or ["0"])]
    trigger = threading.Event()
    if args.voice:
        threading.Thread(target=voice_listener, args=(trigger,), daemon=True).start()

    print("Ready. SPACE = scan, q = quit.")
    try:
        while True:
            for i, cap in enumerate(caps):
                ok, f = cap.read()
                if ok:
                    cv2.imshow(f"camera {i}", cv2.resize(f, None, fx=0.4, fy=0.4))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                trigger.set()
            if trigger.is_set():
                trigger.clear()
                items = []
                for i, cap in enumerate(caps):
                    frame = grab_sharpest(cap)
                    if frame is not None:
                        items.append((f"cam{i}", frame))
                process(engine, items, out_dir, args.max_side)
    except KeyboardInterrupt:
        pass
    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
