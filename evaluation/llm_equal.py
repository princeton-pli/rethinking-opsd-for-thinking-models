"""Answer equivalence by LLM judgement, replacing hand-written string matching.

WHY. `math_equal` is ~300 lines of normalization plus a SymPy fallback — a 2022-era
proxy for semantic equality. Measured on 50 pool pairs (2026-08-26), it mislabels
10% of our "wrong" exemplars: answers that are mathematically identical but written
differently.

    gold                  graded "wrong"      reality
    (3\\sqrt{7},0)         3\\sqrt{7}           the requested x-value
    x                     p(x)=x              same polynomial
    \\det(A)>0             >0                  same claim
    (f(n)=n,g(n)=1)       (f,g)=(n,1)         same pair
    \\mu_2                 \\{1,-1\\}            same group

Those become negative exemplars: the teacher is shown a CORRECT solution labelled
incorrect. A flash-tier model settles all of them in one batched call.

DESIGN
  * exact/numeric fast path first -- roughly half of all pairs never reach the API
  * batches of 200 (measured: 100% coverage, ~53s per 1000 pairs, ~$0.02/1000)
  * coverage is VERIFIED per batch and missing indices are re-asked. The reply
    format drifts ("1: SAME" at small batches, "1. SAME" at large ones), and a
    reader that silently drops unmatched lines looks exactly like a clean result
    -- the same failure shape as the daemonic-Pool grader bug.
  * on-disk cache keyed by the answer pair, so reruns and the online gate are free
  * falls back to math_equal if the API is unreachable, so nothing hard-depends
    on the network
"""
import json
import os
import re
import time
import urllib.request


def _strip_string(s):
    """Light normalization for the fast path.

    Deliberately does NOT import evaluation.utils/grader at module load: those
    pull in `regex` and a large normalization stack, which need not be installed
    wherever this judge runs (it is driven from a laptop, since the API key and
    outbound access live there rather than on the cluster). The heavy
    math_equal is imported lazily, only if the API is unreachable.
    """
    s = str(s).strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("$", ""), (" ", "")):
        s = s.replace(a, b)
    return s.rstrip(".").strip()


def _math_equal_lazy(a, b):
    try:
        from .grader import math_equal
    except Exception:
        return _strip_string(a) == _strip_string(b)
    return bool(math_equal(a, b, timeout=True))

MODEL = os.environ.get("LLM_EQUAL_MODEL", "google/gemini-3.7-flash")
BATCH = int(os.environ.get("LLM_EQUAL_BATCH", "200"))
CACHE_PATH = os.environ.get("LLM_EQUAL_CACHE", "harvest/llm_equal_cache.json")
VERDICT_RE = re.compile(r"(\d+)\s*[:.]\s*(SAME|DIFFERENT)", re.I)

PROMPT_HEAD = (
    "For each numbered pair below, decide whether the two mathematical answers denote "
    "the SAME value/object, ignoring formatting, notation, variable-naming and phrasing "
    "differences. A bare value and a value stated as an equation (e.g. 'x' vs 'p(x)=x') "
    "are the SAME. A point and one of its coordinates are DIFFERENT unless the question "
    "clearly asked for that coordinate.\n\n"
    "Reply with one line per pair, exactly: `<number>: SAME` or `<number>: DIFFERENT`. "
    "Answer every pair, in order.\n\n"
)


def _key_from_zshrc():
    path = os.path.expanduser("~/.zshrc")
    if not os.path.exists(path):
        return os.environ.get("OPENROUTER_API_KEY")
    for line in open(path):
        m = re.search(r'OPENROUTER_API_KEY\s*=\s*["\']?([A-Za-z0-9_\-\.]+)', line)
        if m:
            return m.group(1)
    return os.environ.get("OPENROUTER_API_KEY")


def _fast_path(a, b):
    """Return True/False when the pair is decidable without the API, else None."""
    sa, sb = _strip_string(a), _strip_string(b)
    if sa == sb:
        return True
    try:
        if abs(float(sa) - float(sb)) < 1e-9:
            return True
        return False  # both numeric and unequal
    except (ValueError, TypeError):
        return None


class LLMEqual:
    def __init__(self, cache_path=CACHE_PATH, model=MODEL, batch=BATCH):
        self.model, self.batch, self.cache_path = model, batch, cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path))
            except Exception:
                self.cache = {}
        self.key = _key_from_zshrc()
        self.stats = {"fast": 0, "cached": 0, "api": 0, "fallback": 0}

    def _call(self, prompt, max_tokens):
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=300))
        msg = out["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or ""

    def _ask(self, pairs):
        """pairs: list of (a, b). Returns {index: bool} for whatever came back."""
        items = "\n".join(f'{i+1}. A = "{a}"   B = "{b}"' for i, (a, b) in enumerate(pairs))
        reply = self._call(PROMPT_HEAD + items, max_tokens=40 * len(pairs) + 1000)
        got = {}
        for m in VERDICT_RE.finditer(reply):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(pairs):
                got[idx] = m.group(2).upper() == "SAME"
        return got

    def batch_equal(self, pairs):
        """pairs: list of (gold, candidate). Returns list of bools, same order."""
        result = [None] * len(pairs)
        todo = []
        for i, (a, b) in enumerate(pairs):
            fast = _fast_path(a, b)
            if fast is not None:
                result[i] = fast
                self.stats["fast"] += 1
                continue
            ck = json.dumps([str(a), str(b)])
            if ck in self.cache:
                result[i] = self.cache[ck]
                self.stats["cached"] += 1
                continue
            todo.append(i)

        for start in range(0, len(todo), self.batch):
            idxs = todo[start:start + self.batch]
            sub = [pairs[i] for i in idxs]
            got = {}
            for attempt in range(3):
                missing = [j for j in range(len(sub)) if j not in got]
                if not missing:
                    break
                try:
                    fresh = self._ask([sub[j] for j in missing])
                except Exception as e:
                    print(f"  llm_equal API error ({e}); retry {attempt+1}/3")
                    time.sleep(3)
                    continue
                for local, verdict in fresh.items():
                    got[missing[local]] = verdict
            for j, i in enumerate(idxs):
                if j in got:
                    result[i] = got[j]
                    self.cache[json.dumps([str(pairs[i][0]), str(pairs[i][1])])] = got[j]
                    self.stats["api"] += 1
                else:
                    # Never leave a verdict unset: fall back rather than guess.
                    result[i] = _math_equal_lazy(pairs[i][1], pairs[i][0])
                    self.stats["fallback"] += 1
        self.save()
        return result

    def save(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        json.dump(self.cache, open(tmp, "w"))
        os.replace(tmp, self.cache_path)
