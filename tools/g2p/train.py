"""Train the unknown-word guesser: char-to-phone pre-LN transformer, stress fused into vowel tokens, weight EMA.
uv run --with torch --with numpy python tools/g2p/train.py --name base [--misaki tag|plain|none] [--d 256 ...]"""
import argparse, copy, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from data import CONS, VOWELS  # noqa: E402
from metrics import load, score  # noqa: E402
from phones import fuse, unfuse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAD, BOS, EOS = 0, 1, 2
SRC = ["<pad>", "<cmu>", "<misaki>"] + list("'abcdefghijklmnopqrstuvwxyz")
TGT = ["<pad>", "<s>", "</s>"] + CONS + [s + v for s in ("", "ˈ", "ˌ") for v in VOWELS]
SSTOI, TSTOI = {s: i for i, s in enumerate(SRC)}, {s: i for i, s in enumerate(TGT)}
MAX_SRC, MAX_TGT = 40, 40


class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h, s.q, s.k, s.v, s.o = h, nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(s, x, mem, mask=None, causal=False):
        B, T, D = x.shape
        sp = lambda t: t.view(B, -1, s.h, D // s.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sp(s.q(x)), sp(s.k(mem)), sp(s.v(mem)), attn_mask=mask, is_causal=causal)
        return s.o(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(s, d, h, ff, cross, drop):
        super().__init__()
        s.n1, s.sa = nn.LayerNorm(d), Attn(d, h)
        s.cross = cross
        if cross:
            s.n2, s.ca = nn.LayerNorm(d), Attn(d, h)
        s.n3, s.f1, s.f2 = nn.LayerNorm(d), nn.Linear(d, ff), nn.Linear(ff, d)
        s.drop = nn.Dropout(drop)

    def forward(s, x, mem=None, self_mask=None, mem_mask=None, causal=False):
        h = s.n1(x)
        x = x + s.drop(s.sa(h, h, self_mask, causal))
        if s.cross:
            x = x + s.drop(s.ca(s.n2(x), mem, mem_mask))
        return x + s.drop(s.f2(s.drop(F.relu(s.f1(s.n3(x))))))


class G2P(nn.Module):
    def __init__(s, d=256, h=4, enc=3, dec=3, ff=1024, drop=0.1):
        super().__init__()
        s.cfg = dict(d=d, h=h, enc=enc, dec=dec, ff=ff)
        s.src_emb, s.tgt_emb = nn.Embedding(len(SRC), d), nn.Embedding(len(TGT), d)
        s.src_pos, s.tgt_pos = nn.Embedding(MAX_SRC + 1, d), nn.Embedding(MAX_TGT + 2, d)
        s.enc = nn.ModuleList(Block(d, h, ff, False, drop) for _ in range(enc))
        s.dec = nn.ModuleList(Block(d, h, ff, True, drop) for _ in range(dec))
        s.enc_norm, s.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        s.out = nn.Linear(d, len(TGT))
        s.drop = nn.Dropout(drop)
        for n, p in s.named_parameters():
            if p.dim() == 2 and "emb" not in n and "pos" not in n:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(s.src_emb.weight, std=d ** -0.5); nn.init.normal_(s.tgt_emb.weight, std=d ** -0.5)

    def embed(s, x, emb, pos):
        return s.drop(emb(x) * math.sqrt(emb.embedding_dim) + pos.weight[: x.size(1)])

    def encode(s, src):
        pad = (src != PAD)[:, None, None, :]
        x = s.embed(src, s.src_emb, s.src_pos)
        for b in s.enc:
            x = b(x, self_mask=pad)
        return s.enc_norm(x), pad

    def decode(s, mem, pad, tgt):
        x = s.embed(tgt, s.tgt_emb, s.tgt_pos)
        for b in s.dec:
            x = b(x, mem, mem_mask=pad, causal=True)
        return s.out(s.dec_norm(x))

    @torch.no_grad()
    def greedy(s, src):
        mem, pad = s.encode(src)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(MAX_TGT):
            nxt = s.decode(mem, pad, ys)[:, -1].float().argmax(-1)
            nxt = torch.where(done, PAD, nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:].tolist()


def src_ids(word, tag):
    return (([SSTOI[tag]] if tag else []) + [SSTOI[c] for c in word if c in SSTOI])[:MAX_SRC]


def detok(ids):
    out = []
    for i in ids:
        if i in (EOS, PAD):
            break
        if i > EOS:
            out.append(TGT[i])
    return unfuse(out)


def pad_batch(seqs, dev):
    L = max(map(len, seqs))
    return torch.tensor([q + [PAD] * (L - len(q)) for q in seqs], device=dev)


def predict(model, words, tag, dev, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(words), bs):
        with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            out += [detok(r) for r in model.greedy(pad_batch([src_ids(w, tag) for w in words[i:i + bs]], dev))]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--misaki", default="tag", choices=["tag", "plain", "none"])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--enc", type=int, default=3)
    ap.add_argument("--dec", type=int, default=3)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "weights/g2p/data"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda"
    data = Path(a.data)
    out = ROOT / "weights/g2p/runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    tagged = a.misaki == "tag"
    train = []
    for w, p, src in load(data / "train.tsv"):
        mis = src == "misaki-dict"
        if mis and a.misaki == "none":
            continue
        tag = ("<misaki>" if mis else "<cmu>") if tagged else None
        train.append((src_ids(w, tag), [BOS] + [TSTOI[t] for t in fuse(p)] + [EOS]))
    train = [t for t in train if len(t[1]) <= MAX_TGT + 1]
    held = {k: load(data / f"{k}.tsv") for k in ("dev", "test", "gap")}
    infer_tag = "<cmu>" if tagged else None
    model = G2P(a.d, a.heads, a.enc, a.dec, a.ff, a.drop).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"{a.name}: params {n / 1e6:.2f}M train {len(train)} {vars(a)}", flush=True)
    decay, no_decay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() == 2 and "emb" not in nm and "pos" not in nm else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.98))
    per_epoch = len(train) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * per_epoch, pct_start=0.05)
    t0, best, step = time.time(), -1, 0
    log = open(out / "log.txt", "w")
    for ep in range(a.epochs):
        random.shuffle(train)
        chunks = [sorted(train[i:i + a.bs * 50], key=lambda t: len(t[1])) for i in range(0, len(train), a.bs * 50)]
        batches = [c[i:i + a.bs] for c in chunks for i in range(0, len(c), a.bs) if len(c[i:i + a.bs]) == a.bs]
        random.shuffle(batches)
        tot = 0
        for b in batches:
            src, tgt = pad_batch([s for s, _ in b], dev), pad_batch([t for _, t in b], dev)
            with torch.autocast("cuda", torch.bfloat16):
                mem, pad = model.encode(src)
                logits = model.decode(mem, pad, tgt[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, len(TGT)), tgt[:, 1:].reshape(-1), ignore_index=PAD,
                                   label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1; tot += loss.item()
            k = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - k)
        if ep % 4 == 3 or ep == a.epochs - 1:
            words = [w for w, _, _ in held["dev"]]
            r = score(predict(ema, words, infer_tag, dev), [p for _, p, _ in held["dev"]])
            line = f"ep {ep + 1} loss {tot / len(batches):.3f} dev {r['exact']:.4f} kok {r['kokoro']:.4f} PER {r['PER']:.4f} {time.time() - t0:.0f}s"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            if r["kokoro"] >= best:
                best = r["kokoro"]
                torch.save({"model": ema.state_dict(), "cfg": ema.cfg, "tag": infer_tag, "src": SRC, "tgt": TGT}, out / "model.pt")
    ck = torch.load(out / "model.pt")
    ema.load_state_dict(ck["model"])
    res = {}
    for k, rows in held.items():
        hyps = predict(ema, [w for w, _, _ in rows], infer_tag, dev)
        res[k] = score(hyps, [p for _, p, _ in rows])
        with open(out / f"pred_{k}.tsv", "w") as f:
            f.writelines(f"{w}\t{' '.join(p)}\t{' '.join(h)}\n" for (w, p, _), h in zip(rows, hyps))
    res["params"] = n
    res["mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
