"""
Rollout-budget sweep: truncate long generations and re-grade at smaller caps.

Every model is evaluated once at a 38,912-token generation cap; the smaller
budget points (4,096 / 8,192 / 16,384 / 32,768) in the budget-curve figures
are produced OFFLINE by truncating each generated response to the budget and
re-grading the truncated text. A response whose token count already fits the
budget keeps its recorded is_correct; otherwise the first `budget` tokens are
decoded and re-graded with the task's grader.

Usage:
    python run_budget_sweep.py \
        --file_path checkpoints/<run>/eval/aime24_38912/<merged>.jsonl \
        --tokenizer_path checkpoints/<run> \
        [--budgets 4096 8192 16384 32768] [--dataset aime24]

Writes metrics_budget<N>.json next to the input file (same accuracy/pass@k
schema as run_eval.py, plus response-length stats used by the length panels).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from run_eval import load_jsonl, unbiased_pass_at_k
from evaluation.tasks import get_task

DEFAULT_BUDGETS = [4096, 8192, 16384, 32768]


def get_tokenizer(path: str):
    from transformers import AutoTokenizer

    kwargs = {
        "trust_remote_code": True,
        "use_fast": True,
    }
    tokenizer_config = os.path.join(path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config):
        try:
            with open(tokenizer_config) as f:
                config = json.load(f)
            # Some checkpoints store extra_special_tokens as a list, which
            # AutoTokenizer rejects; passing an empty dict overrides it.
            if isinstance(config.get("extra_special_tokens"), list):
                kwargs["extra_special_tokens"] = {}
        except json.JSONDecodeError:
            pass
    return AutoTokenizer.from_pretrained(path, **kwargs)


def decode_tokens(tokenizer, token_ids):
    try:
        return tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--file_path", type=str, required=True,
                        help="Merged generations JSONL from the 38,912-token eval")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Checkpoint/model dir whose tokenizer produced the generations")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset/task name detection")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write metrics_budget<N>.json (default: next to input)")
    args = parser.parse_args()

    data = list(load_jsonl(args.file_path))
    if not data:
        raise SystemExit(f"Error: {args.file_path} is empty.")

    dataset_name = args.dataset or data[0].get("dataset", "unknown")
    task = get_task(dataset_name)
    print(f"Dataset: {dataset_name} -> {task.__class__.__name__}; "
          f"{len(data)} rows; budgets {args.budgets}")

    tokenizer = get_tokenizer(args.tokenizer_path)
    output_dir = args.output_dir or os.path.dirname(args.file_path)
    os.makedirs(output_dir, exist_ok=True)

    # correctness[budget][question_id] -> list of bools; lengths[budget] -> list of ints
    correctness = {b: defaultdict(list) for b in args.budgets}
    lengths = {b: [] for b in args.budgets}
    grade_errors = 0

    for row in tqdm(data, desc="truncating + regrading"):
        response = row.get("response", "") or ""
        token_ids = tokenizer.encode(response, add_special_tokens=False)
        token_count = len(token_ids)
        qid = task.get_question_id(row)

        for budget in args.budgets:
            if token_count <= budget:
                is_correct = bool(row.get("is_correct", False))
            else:
                truncated = decode_tokens(tokenizer, token_ids[:budget])
                try:
                    gold = task.get_gold(row)
                    grade_result = task.grade(truncated, gold, row)
                    is_correct = bool(grade_result.get("is_correct", False))
                except Exception:
                    is_correct = False
                    grade_errors += 1
            correctness[budget][qid].append(is_correct)
            lengths[budget].append(min(token_count, budget))

    if grade_errors:
        print(f"WARNING: {grade_errors} truncated responses raised grading errors "
              "(counted as incorrect).")

    for budget in args.budgets:
        grouped = correctness[budget]
        results_list = list(grouped.values())
        n_samples = len(results_list[0])
        total_correct = sum(sum(g) for g in grouped.values())
        total_gens = sum(len(g) for g in grouped.values())

        metrics = {
            "budget": budget,
            "accuracy": total_correct / total_gens if total_gens > 0 else 0.0,
        }
        for k in [1, 2, 4, 8, 16, 32, 64, 128]:
            if k <= n_samples:
                metrics[f"pass@{k}"] = unbiased_pass_at_k(results_list, k)
        metrics["mean_response_tokens"] = float(np.mean(lengths[budget]))
        metrics["median_response_tokens"] = float(np.median(lengths[budget]))

        out_path = os.path.join(output_dir, f"metrics_budget{budget}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"budget={budget:>6}: accuracy={metrics['accuracy']:.4f} "
              f"pass@{n_samples}={metrics.get(f'pass@{n_samples}', float('nan')):.4f} "
              f"-> {out_path}")


if __name__ == "__main__":
    main()
