# Evaluation task registry and graders: AIME24 / AIME25 (MathTask), HMMT25
# (HMMTTask), and the held-out Countdown split (CountdownTask).
#
# Math answer extraction and equivalence checking come from utils.py /
# grader.py, which are adapted from
# https://github.com/TianHongZXY/RLVR-Decomposed.

import os
import re
from typing import Dict, Any, List, Union
from collections import Counter
from datasets import load_dataset, load_from_disk

from .utils import extract_answer_math, strip_string
from .grader import math_equal


DATASET_REGISTRY = {
    "aime24": {
        "dataset": "data/eval/aime24.parquet",
        "split": "train",
        "task_type": "math"
    },
    "aime25": {
        "dataset": "data/eval/aime25.parquet",
        "split": "train",
        "task_type": "math"
    },
    "hmmt25": {
        "dataset": "data/eval/hmmt25.parquet",
        "split": "train",
        "task_type": "hmmt"
    },
    "countdown": {
        "dataset": "data/raw/countdown_500.parquet",
        "split": "train",
        "task_type": "countdown"
    },
}


class BaseTask:
    """Base class for all dataset tasks."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def load_data(self):
        """
        Loads data based on the config 'dataset' field.
        Handles HF Hub, JSONL, and Parquet automatically.
        """
        path = self.config.get("dataset")
        split = self.config.get("split", "test")

        print(f"  - Loading data from: {path} (split={split})")

        if path and os.path.exists(path):
            if path.endswith(".parquet"):
                return load_dataset("parquet", data_files=path, split='train') # 'split' arg for parquet is often ignored or defaults to train
            elif path.endswith(".jsonl") or path.endswith(".json"):
                return load_dataset("json", data_files=path)
            elif os.path.exists(f"{path}/{split}"):
                return load_from_disk(f"{path}/{split}")
            elif os.path.isdir(path):
                return load_from_disk(path)
        # 2. Default to HuggingFace Hub
        try:
            ds = load_dataset(path, split=split)
            check_cols = ds[list(ds.keys())[0]].column_names if hasattr(ds, "keys") else ds.column_names
            if "prompt" in check_cols:
                ds = ds.rename_column("prompt", "question")
            return ds
        except Exception as e:
            print(f"    CRITICAL ERROR: Could not load dataset '{path}'. Details: {e}")
            return []

    def format_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """
        Returns:
            - A list of dicts (messages) for chat templating.
            - A string if the prompt is already formatted or raw completion is needed.
        """
        raise NotImplementedError

    def get_gold(self, item: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def grade(self, response: str, gold: Any, item: Dict[str, Any]):
        raise NotImplementedError

    def grading_mode(self) -> str:
        return "inline"

    def build_result_entry(
        self,
        raw_item: Dict[str, Any],
        standard_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        result_entry = dict(raw_item)
        result_entry.pop("datapoint", None)
        result_entry.update(standard_fields)
        return result_entry

    def get_question_id(self, item: Dict[str, Any]) -> Any:
        return item.get("question_id")

    def grade_file(self, file_path: str, regrade: bool = False) -> str:
        return file_path


class CountdownTask(BaseTask):
    _FRAC_RE = re.compile(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}')

    def _last_boxed_content(self, text: str) -> str:
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

    def format_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        data = item.get("datapoint") if isinstance(item.get("datapoint"), dict) else item

        allowed_nums = (
            item.get("datapoint_x") or       # Parquet flattened
            item.get("datapoint_nums") or    # Parquet flattened
            data.get("x") or                 # Nested
            data.get("nums")                 # Nested
        )

        if allowed_nums is None:
            if "prompt" in item:
                return str(item["prompt"])
            return str(item)

        content = (
            item.get("datapoint_input_text") or
            data.get("input_text", "")
        )

        return [
            {"role": "user", "content": content}
        ]

    def get_gold(self, item: Dict[str, Any]) -> Any:
        data = item.get("datapoint") if isinstance(item.get("datapoint"), dict) else item

        # Support flattened 'datapoint_y' AND nested 'y'
        return (
            item.get("datapoint_y") or
            item.get("datapoint_target") or
            data.get("y") or
            data.get("target")
        )

    def grade(self, response: str, gold: Any, item: Dict[str, Any]):
        clean_resp = response.replace("$", "").replace(" ", "")
        boxed = self._last_boxed_content(clean_resp)

        if boxed:
            expression = boxed
        else:
            expression = clean_resp

        # Keep only the expression side of "expr = result". The non-boxed
        # branch always did this; the boxed branch did not, so a response
        # boxing "92+5-66=31" failed the character-class guard below (which
        # forbids "=") and scored 0 regardless of the arithmetic -- while the
        # identical solution boxed as "92-66+5" scored 1. Measured 2026-08-27
        # on 600 ungated Qwen3-4B rollouts: 16.5% box the "expr = result"
        # form, giving a 14.5% false-negative rate over all rollouts (64.4%
        # among graded-wrong) and understating true accuracy ~15 points
        # (77.5% -> 92.0%).
        expression = expression.split("=")[0].strip()

        while r'\frac' in expression or r'\dfrac' in expression or r'\tfrac' in expression:
            new_expr = self._FRAC_RE.sub(r'((\1)/(\2))', expression)
            if new_expr == expression:
                break
            expression = new_expr

        expression = expression.replace(r'\times', '*').replace(r'\cdot', '*')
        expression = expression.replace(r'\div', '/')
        expression = expression.replace(r'\left', '').replace(r'\right', '')
        expression = expression.replace('{', '(').replace('}', ')')

        score = 0
        try:
            if not re.match(r'^[\d\+\-\*\/\(\)\.]+$', expression):
                return {"is_correct": score}
            val = eval(expression)
            if not abs(val - float(gold)) < 1e-5:
                return {"is_correct": score}
        except:
            return {"is_correct": score}

        data = item.get("datapoint") if isinstance(item.get("datapoint"), dict) else item

        sources = [
            item.get("datapoint_x"),
            item.get("datapoint_nums"),
            data.get("x"),
            data.get("nums")
        ]
        allowed_nums = next((x for x in sources if x is not None), [])

        used_nums_str = re.findall(r'\d+(?:\.\d+)?', expression)
        used_nums = [float(n) for n in used_nums_str]
        available = [float(x) for x in allowed_nums]
        score = Counter(used_nums) == Counter(available)

        return {"is_correct": score}


class MathTask(BaseTask):
    def format_prompt(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        content = item.get("problem", item.get("question", ""))
        return [{"role": "user", "content": content}]

    def get_gold(self, item: Dict[str, Any]) -> Any:
        answer = item.get("answer", "")
        solution = item.get("solution", "")
        if answer and 'boxed' in str(answer):
            return extract_answer_math(answer)
        if answer:
            return strip_string(str(answer))
        if solution and 'boxed' in str(solution):
            return extract_answer_math(solution)
        return extract_answer_math(solution)

    def grade(self, response: str, gold: Any, item: Dict[str, Any]):
        pred = extract_answer_math(response)
        score = math_equal(pred, gold, timeout=True)
        return {"is_correct": score}


class HMMTTask(BaseTask):
    """Task for HMMT (Harvard-MIT Math Tournament) problems.

    Unlike MathTask, the gold answers are raw LaTeX (not wrapped in \\boxed{})
    so we skip extract_answer_math on the gold side. Also handles multi-value
    comma-separated answers (e.g. problem 10) with order-agnostic matching.
    """
    def format_prompt(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        content = item.get("problem", "")
        return [{"role": "user", "content": content}]

    def get_gold(self, item: Dict[str, Any]) -> Any:
        # Gold answers are already clean LaTeX; just normalize whitespace/formatting
        raw = item.get("answer", "")
        return strip_string(raw)

    def _is_multi_value(self, raw_answer: str) -> bool:
        """Check if the raw (pre-normalized) answer contains comma-separated values."""
        return ", " in raw_answer

    def grade(self, response: str, gold: Any, item: Dict[str, Any]):
        pred = extract_answer_math(response)

        # Handle multi-value answers (comma-separated, e.g. "val1, val2")
        # Check against the raw answer since strip_string removes spaces
        raw_answer = item.get("answer", "")
        if self._is_multi_value(raw_answer):
            gold_parts = [strip_string(g.strip()) for g in raw_answer.split(",")]
            pred_parts = [strip_string(p.strip()) for p in str(pred).split(",")]
            if len(gold_parts) != len(pred_parts):
                return {"is_correct": False}
            # Order-agnostic set matching
            matched = [False] * len(gold_parts)
            for pp in pred_parts:
                for i, gp in enumerate(gold_parts):
                    if not matched[i] and math_equal(pp, gp, timeout=True):
                        matched[i] = True
                        break
            return {"is_correct": all(matched)}

        score = math_equal(pred, gold, timeout=True)
        return {"is_correct": score}


def get_task(name_or_path: str) -> BaseTask:
    """
    Returns a Task instance.
    If the name is in DATASET_REGISTRY, it uses that config.
    Otherwise, it assumes 'name_or_path' is a direct path/HF-ID and guesses the task type.
    """
    # Strip a trailing token-budget suffix, e.g. "aime24_38912" -> "aime24"
    name_or_path = re.sub(r'_\d+$', '', name_or_path)
    if name_or_path in DATASET_REGISTRY:
        config = DATASET_REGISTRY[name_or_path]
        task_type = config.get("task_type", "math")
    else:
        config = {"dataset": name_or_path, "split": "test"}

        d_lower = name_or_path.lower()
        if "countdown" in d_lower: task_type = "countdown"
        elif "hmmt" in d_lower: task_type = "hmmt"
        else: task_type = "math"

    if task_type == "countdown": return CountdownTask(config)
    if task_type == "hmmt": return HMMTTask(config)

    return MathTask(config)
