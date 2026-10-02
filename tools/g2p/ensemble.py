"""Greedy decoding with log-probs averaged over several runs: scores dev/test/gap, and with --label writes the
ensemble's readings of the train words as a distillation set (same columns as train.tsv, misaki rows keep their tag).
uv run --with torch --with numpy python tools/g2p/ensemble.py run1 run2 ... [--label out.tsv]"""
import argparse, json, sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from metrics import load, score  # noqa: E402
from train import BOS, EOS, MAX_TGT, PAD, ROOT, G2P, detok, pad_batch, src_ids  # noqa: E402


@torch.no_grad()
def greedy(models, src):
    enc = [m.encode(src) for m in models]
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(MAX_TGT):
        lp = sum(m.decode(mem, pad, ys)[:, -1].float().log_softmax(-1) for m, (mem, pad) in zip(models, enc))
        nxt = torch.where(done, PAD, lp.argmax(-1))
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    return ys[:, 1:].tolist()


def predict(models, words, tags, bs=1024):
    out = []
    for i in range(0, len(words), bs):
        src = pad_batch([src_ids(w, t) for w, t in zip(words[i:i + bs], tags[i:i + bs])], "cuda")
        with torch.autocast("cuda", torch.bfloat16):
            out += [detok(r) for r in greedy(models, src)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--label")
    a = ap.parse_args()
    models = []
    for r in a.runs:
        ck = torch.load(ROOT / "weights/g2p/runs" / r / "model.pt")
        m = G2P(**ck["cfg"], drop=0.0).cuda().eval()
        m.load_state_dict(ck["model"])
        models.append(m)
    data = ROOT / "weights/g2p/data"
    res = {}
    for k in ("dev", "test", "gap"):
        rows = load(data / f"{k}.tsv")
        res[k] = score(predict(models, [w for w, _, _ in rows], ["<cmu>"] * len(rows)), [p for _, p, _ in rows])
    print(json.dumps(res), flush=True)
    if a.label:
        rows = [l.split("\t") for l in open(data / "train.tsv").read().splitlines()]
        tags = ["<misaki>" if r[2] == "misaki-dict" else "<cmu>" for r in rows]
        hyps = predict(models, [r[0] for r in rows], tags)
        same = sum(h == r[1].split() for h, r in zip(hyps, rows))
        print(f"teacher agrees with gold on {same / len(rows):.4f} of train", flush=True)
        with open(a.label, "w") as f:
            f.writelines("\t".join([r[0], " ".join(h)] + r[2:]) + "\n" for r, h in zip(rows, hyps) if h)


if __name__ == "__main__":
    main()
