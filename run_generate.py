import os
import json
import argparse
import glob
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from evaluation.tasks import get_task

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--use_r1_style_prompt", action="store_true")
    parser.add_argument("--r1_token", type=str, default="<|begin_of_thought|>")
    parser.add_argument("--enable_thinking", type=str, default="default",
                        help="Enable thinking mode in chat template: 'true', 'false', or 'default' (model decides)")

    
    # Sharding args
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--no_grade", action="store_true",
                        help="Skip inline correctness grading. math_equal implements its timeout by "
                             "forking the worker, and forking a process that holds a vLLM CUDA context "
                             "can deadlock: that stalled shard 0 of job 12413497 for 2h20m and got the "
                             "whole 8-GPU job killed by della's 90-min idle-GPU watchdog. Generate on "
                             "GPU with this flag, then grade on CPU via data_tools/grade_jsonl.py.")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Do not let shard 0 wait for all shards and merge.")
    
    return parser.parse_args()

def run_merge(args, final_filename):
    """Merges all shard files into the final JSONL."""
    search_pattern = f"{final_filename}.shard_*"
    shard_files = [f for f in glob.glob(search_pattern) if not f.endswith(".partial")]
    
    if not shard_files:
        print("No completed shards found to merge.")
        return

    print(f"Supervisor: Merging {len(shard_files)} files...")
    all_data = []
    for s_file in shard_files:
        with open(s_file, 'r') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    all_data.sort(key=lambda x: x.get("question_id", 0))

    print(f"Writing {len(all_data)} records to {final_filename}...")
    with open(final_filename, 'w') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
    
    print("Merge complete.")

def main():
    args = parse_args()
    print("Arguments:", args, flush=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset_short = args.dataset #.split(",")[-1].split("/")[-1]
    base_filename = f"{dataset_short}-temp_{args.temperature}-top_p_{args.top_p}{args.comment}.jsonl"
    final_output_path = os.path.join(args.output_dir, base_filename)

    if args.merge_only:
        run_merge(args, final_output_path)
        return

    shard_file_path = f"{final_output_path}.shard_{args.shard_id}"
    temp_shard_path = f"{shard_file_path}.partial"

    if os.path.exists(shard_file_path):
        print(f"Shard {args.shard_id}: Final file exists. Skipping inference.")
    else:
        # Load Data
        print(f"Shard {args.shard_id}: Loading datasets...")
        d_name = args.dataset
        all_items = []
        global_counter = 0

        print(f"  - Loading {d_name}...")
        
        # Identify Task Type
        task_handler = get_task(d_name)
        ds = task_handler.load_data()
        if not ds:
            print(f"    WARNING: No data found for {d_name}. Skipping.")
            # return
        print(f"    Loaded {len(ds)} items.")
        
        for item in ds:
            # Store raw item + metadata. 
            # We do NOT format prompt yet (allows lazy formatting)
            all_items.append({
                "dataset": d_name,
                "raw_item": item,
                "global_id": global_counter,
                "task_handler": task_handler
            })
            global_counter += 1
            
        # Filter, shard, and resume
        existing_ids = set()
        if os.path.exists(temp_shard_path):
            print(f"Shard {args.shard_id}: Resuming from {temp_shard_path}...")
            with open(temp_shard_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        existing_ids.add(data["question_id"])
                    except: continue

        items_to_process = []
        for i, item in enumerate(all_items):
            # Sharding Logic: Round Robin
            if i % args.num_shards == args.shard_id:
                if item["global_id"] not in existing_ids:
                    items_to_process.append(item)

        # Inference
        if items_to_process:
            print(f"Shard {args.shard_id}: Processing {len(items_to_process)} items...")
            
            # Initialize Model
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.num_gpus,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
                trust_remote_code=True
            )
            
            # Gemma 4 uses special tokens (<|channel>, <channel|>) for thinking —
            # keep them so strip_thinking_tokens() can find and remove them during grading.
            model_basename = os.path.basename(args.model_name).lower()
            _is_gemma4 = ("gemma4" in model_basename or "gemma-4" in model_basename)
            _skip_special = not _is_gemma4

            sampling_params = SamplingParams(
                n=args.num_generation,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                skip_special_tokens=_skip_special,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

            # Pre-format prompts using the Task Registry
            formatted_prompts = []
            
            # Helper to keep prompts aligned with items
            for item in items_to_process:
                handler = item["task_handler"]
                prompt_data = handler.format_prompt(item["raw_item"])

                # Gemma 4 doesn't default to \boxed{} answers; append instruction if missing
                if (
                    _is_gemma4
                    and handler.grading_mode() == "inline"
                    and isinstance(prompt_data, list)
                    and prompt_data
                ):
                    last_content = prompt_data[-1].get("content", "")
                    if "boxed" not in last_content:
                        prompt_data[-1]["content"] = last_content.rstrip() + " Return your final response within \\boxed{}."

                # If handler returns a string, it's already formatted (e.g., pre-templated)
                if isinstance(prompt_data, str):
                    text = prompt_data
                    #formatted_prompts.append(prompt_data)
                # If handler returns a list, apply the chat template
                else:
                    chat_template_kwargs = {}
                    if args.enable_thinking != "default":
                        chat_template_kwargs["enable_thinking"] = (args.enable_thinking == "true")
                    text = tokenizer.apply_chat_template(
                        prompt_data,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs
                    )
                    #formatted_prompts.append(text)

                if args.use_r1_style_prompt:
                    text = text + f"{args.r1_token}\n"
                formatted_prompts.append(text)
                
            # Processing Loop
            with open(temp_shard_path, 'a') as f_out:
                for i in tqdm(range(0, len(items_to_process), args.batch_size)):
                    batch_prompts = formatted_prompts[i : i + args.batch_size]
                    batch_items   = items_to_process[i : i + args.batch_size]
                    
                    outputs = llm.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
                    
                    for j, output in enumerate(outputs):
                        item = batch_items[j]
                        handler = item["task_handler"]
                        
                        # Each prompt generates 'n' responses
                        for k, output_obj in enumerate(output.outputs):
                            resp_text = output_obj.text
                            if args.use_r1_style_prompt:
                                resp_text = f"{args.r1_token}\n" + resp_text

                            standard_fields = {
                                "question_id": item["global_id"],
                                "generation_id": k,
                                "dataset": item["dataset"],
                                "prompt": batch_prompts[j],
                                "response": resp_text,
                                "model_name": args.model_name,
                            }

                            if handler.grading_mode() == "inline":
                                gold = handler.get_gold(item["raw_item"])
                                # gold_answer is always written so an offline grader can work
                                # from the JSONL alone; is_correct only when grading inline.
                                standard_fields["gold_answer"] = str(gold)  # stringify for JSON safety
                                if not args.no_grade:
                                    standard_fields.update(handler.grade(resp_text, gold, item["raw_item"]))

                            result_entry = handler.build_result_entry(
                                item["raw_item"],
                                standard_fields,
                            )
                            
                            f_out.write(json.dumps(result_entry) + "\n")
                    
                    # Crash Safety
                    f_out.flush()
                    os.fsync(f_out.fileno())

            os.rename(temp_shard_path, shard_file_path)
            print(f"Shard {args.shard_id}: Done -> {shard_file_path}")
        
        else:
            if os.path.exists(temp_shard_path):
                os.rename(temp_shard_path, shard_file_path)

    # Supervisor check (Worker 0 Only)
    if args.shard_id == 0 and not args.skip_merge:
        print("Supervisor: Checking shards...")
        expected_files = [f"{final_output_path}.shard_{i}" for i in range(args.num_shards)]
        
        # Simple polling loop
        wait_counts = 0
        while True:
            if all(os.path.exists(f) for f in expected_files):
                print("Supervisor: All shards ready. Merging...")
                time.sleep(2)
                run_merge(args, final_output_path)
                break
            
            if wait_counts > 1000: # Safety break (approx 5 hours)
                print("Supervisor: Timeout waiting for shards.")
                break

            time.sleep(20)
            wait_counts += 1

if __name__ == "__main__":
    main()
