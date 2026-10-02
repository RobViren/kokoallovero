"""Target encodings. 'onset' is the lexicon's own phone list (stress marks at the syllable onset). 'fused' attaches the
stress to its vowel (ˈæ is one token) and puts it back at the onset by maximal onset, as kokoro_map.from_kokoro does."""
from data import CONS, VOWELS

ONSETS = {(c,) for c in CONS if c != "ŋ"} | {tuple(o.split()) for o in (
    "l j|ɡ j|d j|t j|p ɹ|p l|p j|b ɹ|b l|b j|t ɹ|t w|d ɹ|d w|k ɹ|k l|k w|k j|ɡ ɹ|ɡ l|ɡ w|f ɹ|f l|f j|θ ɹ|θ w|ʃ ɹ|v j|m j|h j|n j|"
    "s p|s t|s k|s m|s n|s l|s w|s f|s p ɹ|s p l|s p j|s t ɹ|s k ɹ|s k l|s k w|s k j|ʃ m|ʃ n|ʃ l|ʃ w").split("|")}
STRESS = ("ˈ", "ˌ")


def fuse(toks):
    out, pending = [], ""
    for t in toks:
        if t in STRESS:
            pending = t
        elif t in VOWELS:
            out.append(pending + t); pending = ""
        else:
            out.append(t)
    return out


def unfuse(toks):
    out = []
    for t in toks:
        if t[0] in STRESS:
            j = len(out)
            while j > 0 and out[j - 1] in CONS:
                j -= 1
            run = out[j:]
            k = 0 if j == 0 else next((k for k in range(len(run)) if tuple(run[k:]) in ONSETS), len(run))
            out.insert(j + k, t[0]); out.append(t[1:])
        else:
            out.append(t)
    return out
