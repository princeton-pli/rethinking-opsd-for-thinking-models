"""
Download and clean the Ashkchamp/Openthoughts_math_filtered_30K dataset.

Transformations applied:
  1. System messages dropped from the conversations list.
  2. Assistant thought/solution tags remapped:
       <|begin_of_thought|>  ->  <think>
       <|end_of_thought|>    ->  </think>
       <|begin_of_solution|> ->  (newlines, tag removed)
       <|end_of_solution|>   ->  (tag removed)
  3. User messages in conversations: "Return your final response within \\boxed{}."
     moved from the beginning to the end of the question.
  4. Problem column: "Return your final response within \\boxed{}." appended to end.
  5. Dropped columns: messages, system, generated_token_count, Question.
  6. Kept columns: source, problem, solution, conversations, correct, Answer, COT_Reason.

Usage (from the repo root):
    python data_tools/build_openthoughts_30k.py [--output_path PATH]
"""

import argparse
import os

from data_utils import RAW_DATA_DIR
from datasets import load_dataset

COLUMNS_TO_DROP = ["messages", "system", "generated_token_count", "Question"]
BOXED_INSTRUCTION = "Return your final response within \\boxed{}."


def clean_example(example):
    """Clean conversations and append boxed instruction to problem."""
    convs = example["conversations"]

    new_convs = []
    for msg in convs:
        if msg["from"] == "system":
            continue
        elif msg["from"] == "assistant":
            content = msg["value"]
            content = content.replace("<|begin_of_thought|>", "<think>")
            content = content.replace("<|end_of_thought|>", "</think>")
            content = content.replace("<|begin_of_solution|>", "\n\n")
            content = content.replace("<|end_of_solution|>", "")
            new_convs.append({"from": "assistant", "value": content})
        elif msg["from"] == "user":
            content = msg["value"]
            boxed_prefix = "Return your final response within \\boxed{}. "
            if content.startswith(boxed_prefix):
                content = content[len(boxed_prefix):] + " " + boxed_prefix.strip()
            new_convs.append({"from": "user", "value": content})
        else:
            new_convs.append(msg)

    example["conversations"] = new_convs

    problem = example.get("problem", "")
    if not problem.rstrip().endswith(BOXED_INSTRUCTION):
        example["problem"] = problem.rstrip() + " " + BOXED_INSTRUCTION

    return example


def main():
    parser = argparse.ArgumentParser(
        description="Download and clean Ashkchamp/Openthoughts_math_filtered_30K"
    )
    parser.add_argument(
        "--output_path", type=str,
        default=str(RAW_DATA_DIR / "openthoughts_math_filtered_30k.parquet"),
        help="Output parquet file path",
    )
    args = parser.parse_args()

    print("Loading Ashkchamp/Openthoughts_math_filtered_30K ...")
    ds = load_dataset("Ashkchamp/Openthoughts_math_filtered_30K", split="train",
                      trust_remote_code=True)
    print(f"Loaded: {len(ds)} rows, columns: {ds.column_names}")

    drop_cols = [c for c in COLUMNS_TO_DROP if c in ds.column_names]
    ds_clean = ds.map(clean_example, remove_columns=drop_cols)
    print(f"Cleaned: {len(ds_clean)} rows, columns: {ds_clean.column_names}")

    sample = ds_clean[0]
    assert "system" not in ds_clean.column_names
    roles = [m["from"] for m in sample["conversations"]]
    assert "system" not in roles, "System messages should be removed from conversations"
    assert sample["problem"].rstrip().endswith(BOXED_INSTRUCTION), (
        "Problem should end with boxed instruction"
    )
    print("Sanity checks passed.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    ds_clean.to_parquet(args.output_path)
    print(f"\nSaved to {args.output_path}")
    print(f"Final: {len(ds_clean)} rows, columns: {ds_clean.column_names}")
    print("Done!")


if __name__ == "__main__":
    main()
