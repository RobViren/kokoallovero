"""Phone inventory of the lexicon dialect and its tokenizer, shared by the guesser tools.

Training data is {train,dev,test}.tsv under weights/g2p/data: word, space-separated phones, source tag per line. The
shipped guesser was trained on a private lexicon, with dev/test split by stem group so inflections cannot leak."""
VOWELS = "i ɪ ɛ æ ʌ ə u ʊ ɔ ɑ eɪ oʊ aɪ aʊ ɔɪ ɝ ɚ".split()
CONS = "p b t d k ɡ tʃ dʒ f v θ ð s z ʃ ʒ h m n ŋ l ɹ w j".split()
PHONES = sorted(VOWELS + CONS, key=len, reverse=True)


def phone_tokens(ipa):
    s, out = ipa.replace(" ", ""), []
    while s:
        if s[0] in "ˈˌ":
            out.append(s[0]); s = s[1:]; continue
        p = next((p for p in PHONES if s.startswith(p)), None)
        if p is None:
            return None
        out.append(p); s = s[len(p):]
    return out
