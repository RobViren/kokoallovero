"""spaCy en_core_web_sm PTB tags over a text file, one passage per line: the training data for train_tagger.py.
The shipped tagger was trained on a private corpus, normalized the way src/g2p normalizes text before tagging.
Usage: uv run --with spacy tools/tag_corpus.py texts.txt spacy.jsonl   (needs the en_core_web_sm model)"""
import json, sys
import spacy
texts = open(sys.argv[1]).read().splitlines()
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer", "senter"])
with open(sys.argv[2], "w") as f:
    for i, doc in enumerate(nlp.pipe(texts, batch_size=512, n_process=12)):
        toks = [(t.text, t.tag_) for t in doc if not t.is_space]
        f.write(json.dumps({"id": i, "doc": i, "w": [a for a, _ in toks], "t": [b for _, b in toks]}) + "\n")
