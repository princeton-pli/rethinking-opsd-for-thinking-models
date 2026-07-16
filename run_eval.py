import argparse
import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from evaluation.tasks import get_task

def load_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def unbiased_pass_at_k(results_per_question, k):
    """
    Calculates the unbiased pass@k metric.
    results_per_question: list of lists, where each inner list contains booleans (correct/incorrect) for one question.
    """
    total_pass_k_prob = 0
    total_questions = 0
    
    for labels in results_per_question:
        n = len(labels) # Total samples generated for this question
        c = sum(labels) # Number of correct samples
        
        if n < k:
            continue # Skip if we don't have enough samples for k
            
        # Apply unbiased pass@k formula
        if n - c < k:
            prob = 1.0
        else:
            # prod(1 - k / i) for i in range(n - c + 1, n + 1)
            # This calculates the probability that *none* of the top k are correct, then subtracts from 1
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
            
        total_pass_k_prob += prob
        total_questions += 1
    
    return total_pass_k_prob / total_questions if total_questions > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the generations JSONL file")
    parser.add_argument("--regrade", action="store_true", help="Force re-grading using tasks.py logic instead of trusting the file's 'label'")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset/task name detection")
    args = parser.parse_args()

    print(f"Loading results from {args.file_path}...")
    data = list(load_jsonl(args.file_path))
    
    if not data:
        print("Error: File is empty.")
        return

    # Determine dataset type from the first entry to pick the right Task
    first_entry = data[0]
    dataset_name = args.dataset or first_entry.get("dataset", "unknown")
    task_handler = get_task(dataset_name)
    print(f"Detected dataset: {dataset_name} -> Using task handler: {task_handler.__class__.__name__}")

    file_graded = task_handler.grading_mode() == "file"
    if file_graded:
        graded_file_path = task_handler.grade_file(args.file_path, regrade=args.regrade)
        if graded_file_path != args.file_path:
            print(f"Loading graded results from {graded_file_path}...")
            args.file_path = graded_file_path
            data = list(load_jsonl(args.file_path))
            if not data:
                print("Error: Graded file is empty.")
                return

    # --- 1. Group Data by Question ID ---
    grouped_data = defaultdict(list)
    
    print("Grouping and processing...")
    for item in tqdm(data):
        qid = task_handler.get_question_id(item)
        
        # Grading Logic
        if not file_graded and (args.regrade or "is_correct" not in item):
            # Use the registry to grade on the fly
            gold = task_handler.get_gold(item) # Helper to extract gold from saved item structure
            resp = item.get("response", "")
            
            grade_result = task_handler.grade(resp, gold, item)
            item["gold_answer"] = str(gold)  # Update stored gold with corrected value
            is_correct = grade_result["is_correct"]
            item["is_correct"] = is_correct
            item.update(grade_result)
        else:
            # Trust the existing label
            if "is_correct" not in item:
                raise KeyError(
                    f"Missing is_correct for question_id={qid}. "
                    "File-graded tasks should be graded before metrics are computed."
                )
            is_correct = bool(item["is_correct"])
            
        grouped_data[qid].append(is_correct)

    if args.regrade and not file_graded:
        temp_path = args.file_path + ".tmp"
        with open(temp_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        os.replace(temp_path, args.file_path) # Atomic swap
        print(f"Saving regraded results back to {args.file_path}...")

    # --- 2. Calculate Metrics ---
    # Determine N (samples per question) based on the first group
    n_samples = len(list(grouped_data.values())[0])
    print(f"Detected N={n_samples} samples per question (on average).")

    metrics = {}
    
    # Accuracy (Pass@1 equivalent for greedy, or average correctness)
    total_correct = sum(sum(g) for g in grouped_data.values())
    total_gens = sum(len(g) for g in grouped_data.values())
    metrics["accuracy"] = total_correct / total_gens if total_gens > 0 else 0.0
    
    # Pass@K
    # We define standard K values, but filter by what is possible given N
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    results_list = list(grouped_data.values())
    
    for k in k_values:
        if k <= n_samples:
            score = unbiased_pass_at_k(results_list, k)
            metrics[f"pass@{k}"] = score

    # --- 3. Output ---
    print("\n" + "="*30)
    print(f"Evaluation Results for {dataset_name}")
    print("="*30)
    print(json.dumps(metrics, indent=2))
    print("="*30)

    # Save metrics file next to input file
    out_path = os.path.join(os.path.dirname(args.file_path), "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main()
