"""Score a word<TAB>ipa prediction file (e.g. export.py's torch reference, identical to the C++ output) against a split.
python3 tools/g2p/score.py weights/g2p/data/test.tsv preds.tsv"""
import json, sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from data import phone_tokens  # noqa: E402
from metrics import load, score  # noqa: E402

rows = load(sys.argv[1])
pred = dict(l.split("\t") for l in open(sys.argv[2]).read().splitlines())
print(json.dumps(score([phone_tokens(pred[w]) or [] for w, _, _ in rows], [p for _, p, _ in rows]), ensure_ascii=False))
