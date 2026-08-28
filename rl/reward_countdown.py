# verl custom reward for grader-RL Countdown episodes (SPEC §5).
#
# Outcome-only: last \boxed{} in the response, graded by the SAME logic as
# evaluation/tasks.py::CountdownTask.grade at commit efe9fd3 (the post-
# b0de1b4 grader: strip at '=' BEFORE the character-class guard, exactly-once
# multiset). VENDORED verbatim below rather than imported (the verl job runs
# outside the opsd repo) and rather than reimplemented (an auditor sharing
# the auditee's blind spot is not an audit -- HANDOFF_2026-08-27 §7; spec
# M1). The 6 regression cases from b0de1b4 run via: python reward_countdown.py
#
# Contract (verl@083da9ab, reward_manager naive.py:85-90):
#   compute_score(data_source, solution_str, ground_truth, extra_info)
# solution_str = RESPONSE ONLY (continuation after our prompt prefix; the
# opening <think> lives in the prompt, so the response usually looks like
# "...rest of thinking</think>\n\nanswer with \boxed{...}").
# ground_truth = str(target); extra_info["nums"] = the instance's numbers.
import re
from collections import Counter

_FRAC_RE = re.compile(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}')


def _last_boxed_content(text):
    last = ""
    marker = r"\boxed{"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        pos = idx + len(marker)
        depth = 1
        i = pos
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[pos:i - 1]
        start = idx + len(marker)
    return last


def grade_countdown(response, gold, allowed_nums):
    """Verbatim port of CountdownTask.grade @ efe9fd3 (fixed grader)."""
    clean_resp = response.replace("$", "").replace(" ", "")
    boxed = _last_boxed_content(clean_resp)
    expression = boxed if boxed else clean_resp
    # Keep only the expression side of "expr = result" (the b0de1b4 fix).
    expression = expression.split("=")[0].strip()

    while r'\frac' in expression or r'\dfrac' in expression or r'\tfrac' in expression:
        new_expr = _FRAC_RE.sub(r'((\1)/(\2))', expression)
        if new_expr == expression:
            break
        expression = new_expr

    expression = expression.replace(r'\times', '*').replace(r'\cdot', '*')
    expression = expression.replace(r'\div', '/')
    expression = expression.replace(r'\left', '').replace(r'\right', '')
    expression = expression.replace('{', '(').replace('}', ')')

    try:
        if not re.match(r'^[\d\+\-\*\/\(\)\.]+$', expression):
            return 0.0
        val = eval(expression)
        if not abs(val - float(gold)) < 1e-5:
            return 0.0
    except Exception:
        return 0.0

    used = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', expression)]
    available = [float(x) for x in allowed_nums]
    return 1.0 if Counter(used) == Counter(available) else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  **kwargs):
    nums = (extra_info or {}).get("nums")
    if nums is None:
        raise ValueError("extra_info['nums'] missing -- exporter bug")
    return grade_countdown(solution_str, ground_truth, list(nums))


if __name__ == "__main__":
    # b0de1b4 regression cases (same semantics as the main grader).
    cases = [
        ("\\boxed{92+5-66=31}", "31", [92, 5, 66], 1.0),
        ("\\boxed{92-66+5}", "31", [92, 5, 66], 1.0),
        ("\\boxed{(4*6)+1 = 25}", "25", [4, 6, 1], 1.0),
        ("\\boxed{4*6+2=26}", "25", [4, 6, 1], 0.0),        # wrong numbers
        ("\\boxed{\\frac{92}{2}+5}", "51", [92, 2, 5], 1.0),
        ("\\boxed{92+5-66=32}", "31", [92, 5, 66], 1.0),    # claimed result ignored
        ("thinking</think>\n\nSo: \\boxed{8/(3-8/3)}", "24", [8, 3, 8, 3], 1.0),
        ("no box at all", "31", [92, 5, 66], 0.0),
        ("\\boxed{92+5}", "31", [92, 5, 66], 0.0),          # subset solution
    ]
    bad = 0
    for resp, gold, nums, want in cases:
        got = compute_score("countdown", resp, gold, {"nums": nums})
        ok = got == want
        bad += not ok
        print(("PASS" if ok else "FAIL"), repr(resp[:40]), got)
    raise SystemExit(1 if bad else print("all reward regression cases pass"))
