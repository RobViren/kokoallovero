"""One line per finished run: params and exact/kokoro/PER on dev, test, gap."""
import json
from pathlib import Path

for p in sorted((Path(__file__).resolve().parents[2] / "weights/g2p/runs").glob("*/result.json")):
    r = json.loads(p.read_text())
    cells = "  ".join(f"{k} {r[k]['exact']:.4f}/{r[k]['kokoro']:.4f}/{r[k]['PER']:.4f}" for k in ("dev", "test", "gap"))
    print(f"{p.parent.name:14s} {r['params'] / 1e6:5.2f}M  {cells}")
