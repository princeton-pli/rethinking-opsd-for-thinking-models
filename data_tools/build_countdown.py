"""
Download jasonrqh/Countdown-CoT-20k and build non-overlapping parquet splits.

Outputs by default:
  - data/raw/countdown_20k.parquet
  - data/raw/countdown_15k.parquet
  - data/raw/countdown_500.parquet

The normalized rows preserve the original columns and add:
  - source_index: original HF row index
  - datapoint_input_text: prompt text
  - datapoint_nums / datapoint_x: input numbers
  - datapoint_y / datapoint_target: target number
  - output_text: full response
  - response_suffix: response text after the closing think delimiter
  - response_answer_only: final boxed expression from response_suffix
  - input_count: number of input numbers

Usage from repo root:
    python data_tools/build_countdown.py
"""

import argparse
import os
import pathlib
import re
from collections import Counter


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
RAW_DATA_DIR = DATA_ROOT / "raw"

THINK_END_RE = re.compile(r"</think>\s*|<\|end_of_thought\|>\s*")
NUMBERS_RE = re.compile(r"Numbers:\s*\[([^\]]*)\]")
TARGET_RE = re.compile(r"Target:\s*(-?\d+(?:\.\d+)?)")


def size_label(n):
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def default_output_path(n, suffix):
    return str(RAW_DATA_DIR / f"countdown_{size_label(n)}{suffix}.parquet")


def response_suffix(response):
    text = response or ""
    match = THINK_END_RE.search(text)
    return text[match.end():] if match else ""


def last_boxed_content(text):
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


def parse_numbers(prompt):
    match = NUMBERS_RE.search(prompt or "")
    if not match:
        return []
    numbers = []
    for part in match.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        numbers.append(int(value) if value.is_integer() else value)
    return numbers


def parse_target(prompt):
    match = TARGET_RE.search(prompt or "")
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def normalize_row(example, index):
    answer = example.get("answer") or {}
    prompt = example.get("prompt") or ""
    nums = answer.get("numbers")
    target = answer.get("target")

    if nums is None:
        nums = parse_numbers(prompt)
    if target is None:
        target = parse_target(prompt)

    response = example.get("response") or ""
    suffix = response_suffix(response)
    boxed = last_boxed_content(suffix)
    return {
        "source_index": index,
        "datapoint_input_text": prompt,
        "datapoint_nums": nums,
        "datapoint_x": nums,
        "datapoint_y": target,
        "datapoint_target": target,
        "output_text": response,
        "response_suffix": suffix,
        "response_answer_only": f"\\boxed{{{boxed}}}" if boxed else suffix,
        "input_count": len(nums),
    }


def print_breakdown(name, ds):
    counts = Counter(ds["input_count"])
    missing_suffix = sum(1 for text in ds["response_suffix"] if not text)
    missing_answer_only = sum(1 for text in ds["response_answer_only"] if not text)
    print(f"{name}: {len(ds)} rows")
    print(f"  input_count breakdown: {dict(sorted(counts.items()))}")
    print(f"  rows with empty response_suffix: {missing_suffix}")
    print(f"  rows with empty response_answer_only: {missing_answer_only}")


def main():
    parser = argparse.ArgumentParser(
        description="Build sampled jasonrqh/Countdown-CoT-20k parquet subsets"
    )
    parser.add_argument("--repo_id", type=str, default="jasonrqh/Countdown-CoT-20k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=15000)
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--full_out", type=str,
                        default=str(RAW_DATA_DIR / "countdown_20k.parquet"))
    parser.add_argument("--train_out", type=str, default=None)
    parser.add_argument("--eval_out", type=str, default=None)
    args = parser.parse_args()

    if args.train_out is None:
        args.train_out = default_output_path(args.train_size, "")
    if args.eval_out is None:
        args.eval_out = default_output_path(args.eval_size, "")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the Hugging Face `datasets` package. "
            "Run it from the project environment that has datasets installed."
        ) from exc

    print(f"Loading {args.repo_id} split={args.split} ...")
    ds = load_dataset(args.repo_id, split=args.split)
    print(f"Loaded {len(ds)} rows, columns: {ds.column_names}")

    ds = ds.map(normalize_row, with_indices=True)
    print_breakdown("full", ds)

    unique_prompts = len(set(ds["prompt"]))
    if unique_prompts < args.train_size + args.eval_size:
        raise ValueError(
            f"Need {args.train_size + args.eval_size} unique prompts, "
            f"but only found {unique_prompts}."
        )

    shuffled = ds.shuffle(seed=args.seed)
    train_ds = shuffled.select(range(args.train_size))
    eval_ds = shuffled.select(range(args.train_size, args.train_size + args.eval_size))

    train_prompts = set(train_ds["prompt"])
    eval_prompts = set(eval_ds["prompt"])
    overlap = train_prompts & eval_prompts
    if overlap:
        raise ValueError(f"Train/eval prompt overlap count: {len(overlap)}")

    os.makedirs(os.path.dirname(args.full_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_out) or ".", exist_ok=True)

    ds.to_parquet(args.full_out)
    train_ds.to_parquet(args.train_out)
    eval_ds.to_parquet(args.eval_out)

    print_breakdown("train", train_ds)
    print_breakdown("eval", eval_ds)
    print(f"Train/eval prompt overlap count: {len(overlap)}")
    print(f"Saved full  -> {args.full_out}")
    print(f"Saved train -> {args.train_out}")
    print(f"Saved eval  -> {args.eval_out}")
    print("Done!")


if __name__ == "__main__":
    main()
