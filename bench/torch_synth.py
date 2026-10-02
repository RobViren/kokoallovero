"""Torch synth time for one phoneme string through ref/kokoro.py, load excluded, after warmup.
Usage: torch_synth.py <phonemes.txt> [threads] [reps]"""
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
from kokoro import Kokoro  # noqa: E402

ps = Path(sys.argv[1]).read_text().strip()
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
torch.set_num_threads(threads)
model = Kokoro.from_hub()
voice = model.load_voice("af_heart")
for _ in range(2):
    audio = model(ps, voice)
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    model(ps, voice)
    times.append(time.perf_counter() - t0)
sec = audio.numel() / 24000
print(f"torch {torch.__version__} threads {threads} phonemes {len(ps)} audio {sec:.2f}s "
      f"min_ms {min(times) * 1e3:.0f} med_ms {statistics.median(times) * 1e3:.0f} rtf {min(times) / sec:.3f}")
