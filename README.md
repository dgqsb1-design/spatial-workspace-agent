# idea-capture

Write an idea anywhere in your room (wall, mirror, cupboard, whiteboard, paper), tell the program to scan, and get it back as a Word file and a `.tex` file.

- Equations become LaTeX you can paste straight into Overleaf.
- Plain notes become text.
- Diagrams and anything vague become a description in words, plus a draft TikZ figure when the sketch is simple.
- Every capture keeps the photo, so nothing is lost if a read is wrong.

It runs entirely on your own computer, costs nothing, and needs no API key, no account and no extra program. One `pip install`, and the model downloads itself on the first run.

## How it works

The reader is [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Apache-2.0), a vision-language model that reads handwriting and maths, run directly in Python through Hugging Face `transformers`. In October 2025 it was the strongest openly licensed model in this size class for reading handwriting and describing sketches in one pass.

## Install

```bash
git clone https://github.com/<dgqsb1-design>/idea-capture
cd idea-capture
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. The first run downloads about 8 GB of model weights from Hugging Face and caches them.

## Use

```bash
python -m idea_capture                        # live camera: SPACE scans, q quits
python -m idea_capture --image photo.jpg      # no camera: scan a saved photo
python -m idea_capture --camera http://192.168.1.20:8080/video   # phone as camera
```

Try `--image` first with a photo of your own handwriting. With several `--camera` options, every camera is checked and only views with writing are saved. A phone running an IP-camera app (IP Webcam, DroidCam and similar) is the easiest way to point a camera at whatever you just wrote on.

Optional voice trigger: `pip install SpeechRecognition pyaudio pocketsphinx`, then add `--voice` and say "scan". It runs offline but is experimental.

## Output

```
notes/
  ideas.docx     one section per capture: photo, LaTeX in monospace, notes, diagram descriptions
  ideas.tex      the same content as LaTeX, appended on every capture
  images/        the enhanced photo of every capture
```

Example `ideas.tex` after one capture:

```latex
% ===== Capture 20251025-153042 (cam0) =====
\begin{align*}
  \frac{dS}{dt} &= k \\
  t &= \frac{1}{k}\,\Delta S
\end{align*}
% [diagram] Arrow from a box labelled S to a box labelled t.
\iffalse % draft TikZ: delete this line and the matching \fi to enable
\begin{figure}[h]\centering
\begin{tikzpicture}
  \node[draw] (S) at (0,0) {$S$};
  \node[draw] (t) at (3,0) {$t$};
  \draw[->] (S) -- (t);
\end{tikzpicture}
\end{figure}
\fi
```

TikZ drafts sit inside `\iffalse ... \fi`, so a wrong guess can never break your compile. Add `\usepackage{amsmath,tikz}` to your preamble.

## Hardware and speed

- A GPU with 10 GB or more of memory: a scan takes a few seconds.
- CPU only, with 16 GB of RAM: it works, but expect a minute or two per scan.
- If memory is tight, pass a smaller model, for example `--model Qwen/Qwen3-VL-2B-Instruct`. Any Qwen-VL model from Hugging Face works the same way.

## Accuracy and privacy

- Handwritten maths is hard for a 4B model. Write large, use a dark marker, and keep the lighting even. Anything it cannot read is marked as "Unclear" instead of guessed.
- Always check the output against the wall before relying on it.
- Nothing leaves your machine after the model has downloaded. Photos are saved only in your `notes/` folder.

## Troubleshooting

- Out of memory: use the smaller model above, or lower `--max-side` (default 1280).
- `Qwen3VLForConditionalGeneration` import error: upgrade with `pip install -U transformers`.
- Camera not opening: try `--camera 1`, or use a phone stream URL.
- Output appears as an "Unclear" block of raw text: the model ignored the format. Scan again, or try a larger model.

## Contributing

Pull requests are welcome. Ideas: a Markdown output format, a region-of-interest selector, a compile check for the LaTeX, and a lighter OCR-only mode for very small machines.

## License

MIT
