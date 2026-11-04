"""quote_match was sped up with a cheap upper-bound check (quick_ratio). The verdict ('exact' / 'approx' / 'none') and every
ratio >= 0.85 must stay identical to the original exhaustive scan; kept here as the reference."""
import difflib
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "prototype"))

from llm_stage import _norm_quote, quote_match  # noqa: E402


def reference_quote_match(quote, texts):
    q = _norm_quote(quote)
    if len(q) < 3:
        return 0.0, "none"
    qwords = q.split()
    best = 0.0
    for t in texts:
        nt = _norm_quote(t)
        if q in nt:
            return 1.0, "exact"
        words = nt.split()
        for k in {max(1, len(qwords) - 1), len(qwords), len(qwords) + 1}:
            for i in range(0, max(1, len(words) - k + 1)):
                best = max(best, difflib.SequenceMatcher(None, q, " ".join(words[i:i + k]), autojunk=False).ratio())
    return best, ("approx" if best >= 0.85 else "none")


WORDS = ("the ego is an centric behaviour a part of our personality it reveals identity self awareness and ability to make "
         "decisions arises from ignorance false assumptions about ourselves contradicts reality acts as barrier invoice total "
         "paid march quarterly report north south east units 4821 129.50 meeting minutes budget approved deadline").split()


def garble(text, rng, rate):
    out = []
    for ch in text:
        r = rng.random()
        if ch.isalpha() and r < rate:
            out.append(rng.choice("aeiourstln"))
        elif ch.isalpha() and r < rate * 1.3:
            continue
        else:
            out.append(ch)
    return "".join(out)


def test_the_faster_quote_match_gives_the_same_verdicts_and_ratios():
    rng = random.Random(11)
    checked = {"exact": 0, "approx": 0, "none": 0}
    for _ in range(120):
        pages = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(30, 90))) for _ in range(rng.randint(1, 3))]
        src = rng.choice(pages).split()
        i = rng.randint(0, max(0, len(src) - 8))
        quote = " ".join(src[i:i + rng.randint(3, 8)])
        kind = rng.choice(["same", "garbled", "garbled", "unrelated"])
        if kind == "garbled":
            quote = garble(quote, rng, rng.choice([0.04, 0.08, 0.2]))
        elif kind == "unrelated":
            quote = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 8)))
        ref, new = reference_quote_match(quote, pages), quote_match(quote, pages)
        assert ref[1] == new[1], (quote, ref, new)
        if ref[1] != "none":
            assert abs(ref[0] - new[0]) < 1e-9, (quote, ref, new)
        else:
            assert new[0] <= ref[0] + 1e-9 and new[0] < 0.85
        checked[ref[1]] += 1
    assert all(checked.values()), checked                  # every verdict was exercised


def test_a_quote_that_is_not_in_a_long_document_is_checked_quickly():
    rng = random.Random(3)
    pages = [" ".join(rng.choice(WORDS) for _ in range(300)) for _ in range(6)]
    start = time.time()
    assert quote_match("shipping address is Pune and the parcel was delivered late on friday", pages)[1] == "none"
    assert time.time() - start < 6.0
