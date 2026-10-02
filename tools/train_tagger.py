"""Averaged perceptron PTB tagger distilled from spaCy en_core_web_sm, exported for src/g2p/tagger.cc.
python tools/train_tagger.py spacy.jsonl [epochs]   (spacy.jsonl from tools/tag_corpus.py)
Writes weights/g2p/tagger.bin and weights/g2p/tagger_test.tsv (held-out sentences, tab-separated tokens)."""
import json
import random
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "weights" / "g2p"
SEP = b"\x1f"
PAD_L, PAD_R = [b"-S2-", b"-S-"], [b"-E-", b"-E2-"]
P1, P2, P1P2, P1W = 30, 31, 32, 33
MIN_COUNT = 2
QSCALE = 1024.0


def lower(w):
    return w.lower()


def shape(w):
    out, last, run = bytearray(), -1, 0
    for b in w:
        c = 88 if 65 <= b <= 90 else 120 if 97 <= b <= 122 else 100 if 48 <= b <= 57 else 117 if b >= 128 else b
        run = run + 1 if c == last else 1
        last = c
        if run <= 4:
            out.append(c)
    return bytes(out)


def static_feats(ws, i):
    """ws: padded lowercase words; i: index into ws (real tokens start at 2). Raw word needed for shape."""
    w, p, pp, n, nn = ws[i], ws[i - 1], ws[i - 2], ws[i + 1], ws[i + 2]
    return [
        (0, b""), (1, w), (2, w[-2:]), (3, w[-3:]), (4, w[-4:]), (5, w[:1]), (6, w[:2]),
        (8, p), (9, p[-3:]), (11, pp), (12, n), (13, n[-3:]), (15, nn),
        (16, p + SEP + w), (17, w + SEP + n),
    ]


def sentence_static(raw):
    ws = PAD_L + [lower(w) for w in raw] + PAD_R
    rs = PAD_L + raw + PAD_R
    out = []
    for i in range(2, len(ws) - 2):
        f = static_feats(ws, i)
        f += [(7, shape(rs[i])), (10, shape(rs[i - 1])), (14, shape(rs[i + 1]))]
        out.append([bytes([t]) + SEP + v for t, v in f])
    return out


def dyn_feats(p1, p2, w):
    return [bytes([P1]) + SEP + p1, bytes([P2]) + SEP + p2, bytes([P1P2]) + SEP + p1 + SEP + p2, bytes([P1W]) + SEP + p1 + SEP + w]


def fnv1a(b):
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def load(path):
    docs = []
    for line in open(path):
        d = json.loads(line)
        if d["w"]:
            docs.append((d["doc"], [w.encode() for w in d["w"]], d["t"]))
    return docs


def split(docs, frac=0.05, seed=0):
    rng = random.Random(seed)
    by = {}
    for i, (doc, _, _) in enumerate(docs):
        by.setdefault(doc, []).append(i)
    test = set()
    for doc in sorted(by):
        idx = by[doc]
        test.update(rng.sample(idx, max(1, round(len(idx) * frac))))
    return [d for i, d in enumerate(docs) if i not in test], [d for i, d in enumerate(docs) if i in test]


class Model:
    def __init__(self, tags, feats):
        self.tags = tags
        self.tid = {t: i for i, t in enumerate(tags)}
        self.fid = {f: i for i, f in enumerate(feats)}
        self.W = np.zeros((len(feats), len(tags)), np.float32)
        self.tot = np.zeros_like(self.W, dtype=np.float64)
        self.ts = np.zeros(self.W.shape, np.int64)
        self.t = 0

    def ids(self, fs):
        return [self.fid[f] for f in fs if f in self.fid]

    def tag_sentence(self, raw, stat_ids, W, gold=None):
        n = len(raw)
        S = np.stack([W[ix].sum(0) for ix in stat_ids]) if n else None
        prev = [b"-S2-", b"-S-"]
        lw = [lower(w) for w in raw]
        out = []
        for i in range(n):
            d = self.ids(dyn_feats(prev[-1], prev[-2], lw[i]))
            s = S[i] + W[d].sum(0)
            g = int(np.argmax(s))
            if gold is not None:
                self.t += 1
                y = self.tid[gold[i]]
                if g != y:
                    ix = np.array(stat_ids[i] + d, np.int64)
                    for c, delta in ((y, 1.0), (g, -1.0)):
                        self.tot[ix, c] += (self.t - self.ts[ix, c]) * self.W[ix, c]
                        self.ts[ix, c] = self.t
                        self.W[ix, c] += delta
                g = y
            out.append(self.tags[g])
            prev.append(self.tags[g].encode())
        return out

    def averaged(self):
        return ((self.tot + (self.t - self.ts) * self.W) / self.t).astype(np.float32)


def train(docs, train_docs, epochs):
    t0 = time.time()
    cnt = Counter()
    train_static = []
    for _, raw, tags in train_docs:
        st = sentence_static(raw)
        train_static.append(st)
        for fs in st:
            cnt.update(fs)
        lw = [lower(w) for w in raw]
        prev = ["-S2-", "-S-"] + tags
        for i in range(len(raw)):
            cnt.update(dyn_feats(prev[i + 1].encode(), prev[i].encode(), lw[i]))
    feats = [f for f, c in cnt.items() if c >= MIN_COUNT]
    tags = sorted({t for _, _, ts in docs for t in ts})
    print(f"{len(cnt)} features, {len(feats)} kept, {len(tags)} tags, {time.time() - t0:.0f}s", flush=True)
    del cnt
    m = Model(tags, feats)
    train_ids = [[m.ids(fs) for fs in st] for st in train_static]
    del train_static
    order = list(range(len(train_docs)))
    rng = random.Random(1)
    for ep in range(epochs):
        rng.shuffle(order)
        t0 = time.time()
        for k in order:
            _, raw, gold = train_docs[k]
            m.tag_sentence(raw, train_ids[k], m.W, gold)
        print(f"epoch {ep} {time.time() - t0:.0f}s", flush=True)
    return tags, feats, m.averaged()


def main():
    """argv: spacy.jsonl, epochs (0 re-exports the cached averaged model), prune (min |weight| in 1/QSCALE units)."""
    docs = load(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    prune = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    train_docs, test = split(docs)
    print(f"train {len(train_docs)} chunks {sum(len(d[1]) for d in train_docs)} toks, test {len(test)} chunks {sum(len(d[1]) for d in test)} toks", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "tagger_avg.npz"
    if epochs:
        tags, feats, A = train(docs, train_docs, epochs)
        np.savez(cache, tags=np.array(tags), feats=np.array(feats, dtype=object), A=A)
    else:
        z = np.load(cache, allow_pickle=True)
        tags, feats, A = list(z["tags"]), list(z["feats"]), z["A"]
    Q = np.round(A * QSCALE).clip(-32767, 32767).astype(np.int16)
    Q[np.abs(Q) < prune] = 0
    keep = np.flatnonzero(np.abs(Q).max(1) > 0)
    m = Model(tags, feats)
    Wq = Q.astype(np.float32)
    preds = [m.tag_sentence(raw, [m.ids(fs) for fs in sentence_static(raw)], Wq) for _, raw, _ in test]
    ok = sum(a == b for (_, _, g), p in zip(test, preds) for a, b in zip(g, p))
    n = sum(len(g) for _, _, g in test)
    print(f"held-out agreement (quantized, prune {prune}) {ok / n:.4f} over {n} tokens", flush=True)
    write(OUT / "tagger.bin", tags, feats, Q, keep)
    with open(OUT / "tagger_test.tsv", "w") as f:
        for (_, raw, g), p in zip(test, preds):
            f.write("\t".join(w.decode() for w in raw) + "\n" + "\t".join(g) + "\n" + "\t".join(p) + "\n")


def write(path, tags, feats, Q, keep):
    """KKTG v1: u32 magic, version, ntags, cap (pow2), nentries; ntags x char[8]; cap x {u32 fp, u32 off};
    nentries x u32 (weight i16 << 16 | last << 8 | tag). fp = hash >> 32 (0 means empty, so 0 maps to 1),
    linear probe from hash & (cap - 1)."""
    cap = 1 << max(4, (int(len(keep) / 0.7) - 1).bit_length())
    fps = np.zeros(cap, np.uint32)
    offs = np.zeros(cap, np.uint32)
    ent = []
    seen = set()
    for r in keep:
        h = fnv1a(feats[r])
        assert h not in seen, "hash collision"
        seen.add(h)
        fp, s = (h >> 32) or 1, h & (cap - 1)
        while fps[s]:
            s = (s + 1) & (cap - 1)
        cs = np.flatnonzero(Q[r])
        fps[s], offs[s] = fp, len(ent)
        ent.extend(((int(Q[r, c]) & 0xFFFF) << 16) | (int(i == len(cs) - 1) << 8) | int(c) for i, c in enumerate(cs))
    with open(path, "wb") as f:
        f.write(struct.pack("<5I", 0x47544B4B, 1, len(tags), cap, len(ent)))
        f.write(b"".join(t.encode().ljust(8, b"\0") for t in tags))
        f.write(np.stack([fps, offs], 1).astype("<u4").tobytes())
        f.write(np.array(ent, "<u4").tobytes())
    print(f"{path}: {len(keep)} features, {len(ent)} weights, cap {cap}, {path.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
