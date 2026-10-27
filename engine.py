"""Local handwriting reader.

Runs Qwen3-VL-4B-Instruct (Apache-2.0) directly inside Python through Hugging Face
`transformers`. No server, no API key, no account, no cost: the weights (about 8 GB)
download once on first run and are cached.

The model is asked for a simple tagged format instead of JSON, because LaTeX is full of
backslashes and small models routinely break JSON escaping (\\frac, \\theta, ...).
"""

import io

PROMPT = r"""This is a photo of handwriting or sketches on a surface in a room (wall, mirror,
cupboard, whiteboard, paper). The author is a theoretical physicist jotting down ideas.
Ignore the surface itself, reflections of the room, glare, and people.
If the writing appears mirror-reversed because it is a reflection, read it as if corrected.

Answer with a sequence of blocks in reading order. Each block starts with a line that
contains only one of these tags:

@@math      complete LaTeX for an equation or a group of aligned equations, including its
            environment (equation*, align*, ...), ready to paste into a .tex file
@@text      plain words and sentences, with inline $...$ for symbols
@@diagram   a description in words of a sketch: its elements, labels, arrows and relations
@@tikz      (directly after a @@diagram block, optional) a minimal tikzpicture environment,
            only if the sketch is simple enough to reproduce
@@unclear   what is there and why it cannot be read, with a best guess if you have one

If there is no writing or drawing at all, answer only: @@none

Transcribe exactly what is written. Do not solve, correct or extend the physics.
No JSON, no code fences, no commentary outside the blocks."""

TAGS = {"math", "text", "diagram", "tikz", "unclear", "none"}


def parse_blocks(raw: str) -> dict:
    """Turn the tagged reply into {"has_writing": bool, "blocks": [...]}."""
    blocks, cur, in_tikz = [], None, False
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("```"):
            continue
        tag = s[2:].strip().lower() if s.startswith("@@") else None
        if tag in TAGS:
            if tag == "tikz":
                if cur is None or cur["type"] != "diagram":
                    cur = {"type": "diagram", "content": [], "tikz": []}
                    blocks.append(cur)
                in_tikz = True
            else:
                cur = {"type": tag, "content": [], "tikz": []}
                blocks.append(cur)
                in_tikz = False
            continue
        if cur is None:  # the model ignored the format: keep what it said
            cur = {"type": "unclear", "content": [], "tikz": []}
            blocks.append(cur)
        cur["tikz" if in_tikz else "content"].append(line)

    out = []
    for b in blocks:
        if b["type"] == "none":
            continue
        content, tikz = "\n".join(b["content"]).strip(), "\n".join(b["tikz"]).strip()
        if content or tikz:
            out.append({"type": b["type"], "content": content, "tikz": tikz})
    return {"has_writing": bool(out), "blocks": out}


class Engine:
    name = "Qwen3-VL (local)"

    def __init__(self, model_id: str | None = None):
        import torch
        from transformers import AutoProcessor

        self.model_id = model_id or "Qwen/Qwen3-VL-4B-Instruct"
        self.model = self.model_id  # shown in progress messages
        print(f"Loading {self.model_id} (the first run downloads about 8 GB) ...")
        if "qwen3" in self.model_id.lower():
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        else:
            from transformers import AutoModelForImageTextToText as ModelClass
        self.model_obj = ModelClass.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.torch = torch
        print(f"Model ready on {self.model_obj.device}.")

    def read(self, jpeg: bytes) -> dict:
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model_obj.device)
        with self.torch.inference_mode():
            ids = self.model_obj.generate(**inputs, max_new_tokens=1500, do_sample=False)
        new_tokens = ids[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return parse_blocks(raw)
