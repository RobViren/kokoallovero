"""SNR of the C++ dumps against torch, stage by stage. floor is torch against itself with the style nudged by one
float32 ulp (same noise), the level below which a difference is numerics rather than a bug.
Usage: parity.py [cpp_dump_dir]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "weights" / "ref"
CPP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "weights" / "cpp"
STAGES = ["bert", "bert_enc", "dur_enc", "durations", "frames", "f0", "n", "text_enc", "dec.encode",
          "dec.decode0", "dec.decode1", "dec.decode2", "dec.decode3", "source", "har", "gen.up0", "gen.noiseconv0",
          "gen.noise0", "gen.res0", "gen.up1", "gen.noiseconv1", "gen.noise1", "gen.res1", "conv_post", "audio"]


def snr(ref, x):
    if ref.shape != x.shape:
        return float("nan")
    err = np.sum((ref.astype(np.float64) - x) ** 2)
    return float("inf") if err == 0 else 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / err)


print(f"{'stage':<15}{'shape':>14}{'snr_db':>9}{'floor_db':>10}{'max_abs':>11}")
for s in STAGES:
    if not (CPP / f"{s}.npy").exists():
        print(f"{s:<15}{'missing':>14}")
        continue
    ref, cpp, ulp = (np.load(d / f"{s}.npy") for d in (REF, CPP, REF / "ulp"))
    cpp = cpp.reshape(cpp.shape if cpp.shape == ref.shape else (-1,))
    m = float(np.max(np.abs(ref - cpp))) if ref.shape == cpp.shape else float("nan")
    print(f"{s:<15}{'x'.join(map(str, ref.shape)):>14}{snr(ref, cpp):>9.1f}{snr(ref, ulp):>10.1f}{m:>11.2e}")
