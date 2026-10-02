"""Kokoro-82M to flat f32 blobs the C++ mmaps. Downloads the checkpoint and voices from hexgrad/Kokoro-82M.
Usage: uv run tools/export_weights.py [voice ...]
Writes weights/{kokoro,voices}.{bin,tsv} (name, offset in floats, shape; every tensor 64-byte aligned) and
weights/vocab.tsv (codepoint, id)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref"))
from kokoro import Kokoro  # noqa: E402


def write(stem, tensors):
    off = 0
    with open(stem.with_suffix(".bin"), "wb") as blob, open(stem.with_suffix(".tsv"), "w") as index:
        for name, t in tensors.items():
            a = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype="<f4")
            pad = -off % 16
            blob.write(bytes(4 * pad))
            off += pad
            index.write(f"{name}\t{off}\t{','.join(map(str, a.shape))}\n")
            blob.write(a.tobytes())
            off += a.size
    print(f"{stem}: {len(tensors)} tensors, {off * 4 / 2**20:.1f} MiB")


model = Kokoro.from_hub()
out = ROOT / "weights"
out.mkdir(exist_ok=True)
write(out / "kokoro", model.state_dict())
write(out / "voices", {v: model.load_voice(v) for v in (sys.argv[1:] or ["af_heart", "am_michael", "bf_emma"])})
(out / "vocab.tsv").write_text("".join(f"{ord(p)}\t{i}\n" for p, i in model.vocab.items()))
