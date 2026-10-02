"""Scores on phone lists in the lexicon dialect. exact: identical with stress at the onset. kokoro: identical after
the map to Kokoro's phoneme set (what Kokoro hears; onset placement no longer matters), which needs the ag2p package
and falls back to exact without it. nostress: segments only. PER: phone edit distance without stress marks over
reference phones."""
import collections, sys
from pathlib import Path

sys.dont_write_bytecode = True
try:
    from ag2p import kokoro_map as km
except ImportError:
    km = None
from data import VOWELS  # noqa: E402

STRESS = ("ˈ", "ˌ")


def load(path):
    return [(r[0], r[1].split(), r[2]) for r in (l.split("\t") for l in open(path).read().splitlines())]


def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def kok(toks):
    try:
        return km.to_kokoro("".join(toks)) if km else "".join(toks)
    except AssertionError:
        return "".join(toks)


def seg(toks):
    return [t for t in toks if t not in STRESS]


def classify(h, r):
    """Coarse class of a miss: stress (segments right), ə/ɪ (the only segment errors are ə vs ɪ), other vowel
    (consonants right), consonant, length."""
    if seg(h) == seg(r):
        return "stress-only"
    hs, rs = seg(h), seg(r)
    if len(hs) != len(rs):
        return "insert/delete"
    diff = [(a, b) for a, b in zip(hs, rs) if a != b]
    if all({a, b} <= {"ə", "ɪ"} for a, b in diff):
        return "ə/ɪ-only"
    if all(a in VOWELS and b in VOWELS for a, b in diff):
        return "vowel-only" if len(diff) == 1 else "vowels-multi"
    return "consonant"


def score(hyps, refs):
    n = len(refs)
    ex = sum(h == r for h, r in zip(hyps, refs))
    kx = sum(kok(h) == kok(r) for h, r in zip(hyps, refs))
    nx = sum(seg(h) == seg(r) for h, r in zip(hyps, refs))
    per = sum(ed(seg(h), seg(r)) for h, r in zip(hyps, refs)) / sum(len(seg(r)) for r in refs)
    cls = collections.Counter(classify(h, r) for h, r in zip(hyps, refs) if kok(h) != kok(r))
    return {"n": n, "exact": round(ex / n, 4), "kokoro": round(kx / n, 4), "nostress": round(nx / n, 4),
            "PER": round(per, 4), "miss": {k: round(v / n, 4) for k, v in cls.most_common()}}
