"""Write a trained guesser (runs/<name>/model.pt) to the flat format src/g2p/guesser.cc reads, plus torch's own fp32
CPU greedy outputs on a word list for the C++ parity check.
uv run --with torch --with numpy python tools/g2p/export.py weights/g2p/runs/<name>/model.pt weights/g2p/guesser.bin [words.tsv out.tsv]

Format, little endian: "KKG2P001"; i32 d heads enc dec ff max_src max_tgt tag; u32 n_src then (u8 len, utf8) per char;
u32 n_tgt then (u8 kind 0 special/1 consonant/2 vowel, u8 stress 0/1 primary/2 secondary, u8 len, utf8 phone);
u32 n_onsets then (u8 n, u8 ids...) per legal onset; u32 n_tensors then (u8 len, name, u8 ndim, i32 dims, f16 data).
Weights are rounded to f16 first, so the torch reference sees exactly what C++ computes with."""
import struct, sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from data import CONS, VOWELS  # noqa: E402
from phones import ONSETS  # noqa: E402
from train import G2P, MAX_SRC, MAX_TGT, SSTOI, detok, src_ids  # noqa: E402


def main():
    ck = torch.load(sys.argv[1], map_location="cpu")
    src, tgt, cfg = ck["src"], ck["tgt"], ck["cfg"]
    tag = src.index(ck["tag"]) if ck["tag"] else -1
    b = bytearray(b"KKG2P001")
    b += struct.pack("<8i", cfg["d"], cfg["h"], cfg["enc"], cfg["dec"], cfg["ff"], MAX_SRC, MAX_TGT, tag)
    b += struct.pack("<I", len(src))
    for s in src:
        e = s.encode() if len(s) == 1 else b""
        b += struct.pack("<B", len(e)) + e
    b += struct.pack("<I", len(tgt))
    for t in tgt:
        st = 1 if t[0] == "ˈ" else 2 if t[0] == "ˌ" else 0
        base = t[1:] if st else t
        kind = 1 if base in CONS else 2 if base in VOWELS else 0
        e = base.encode() if kind else b""
        b += struct.pack("<3B", kind, st, len(e)) + e
    b += struct.pack("<I", len(ONSETS))
    for o in sorted(ONSETS):
        b += struct.pack(f"<{len(o) + 1}B", len(o), *(tgt.index(c) for c in o))
    sd = {k: v.half().float() for k, v in ck["model"].items()}
    b += struct.pack("<I", len(sd))
    for name, t in sd.items():
        t = t.float().contiguous()
        e = name.encode()
        b += struct.pack(f"<B{len(e)}sB{t.dim()}i", len(e), e, t.dim(), *t.shape) + t.half().numpy().tobytes()
    Path(sys.argv[2]).write_bytes(bytes(b))
    print(f"{sys.argv[2]}: {len(b) / 1e6:.2f} MB")
    if len(sys.argv) > 4:
        m = G2P(**cfg, drop=0.0)
        m.load_state_dict(sd)
        m.eval()
        torch.set_num_threads(8)
        words = [l.split("\t")[0] for l in open(sys.argv[3]).read().splitlines()]
        with open(sys.argv[4], "w") as f:
            for i in range(0, len(words), 256):
                ws = words[i:i + 256]
                ids = [src_ids(w, ck["tag"]) for w in ws]
                L = max(map(len, ids))
                for w, r in zip(ws, m.greedy(torch.tensor([x + [0] * (L - len(x)) for x in ids]))):
                    f.write(f"{w}\t{''.join(detok(r))}\n")


if __name__ == "__main__":
    main()
