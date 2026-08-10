#!/usr/bin/env python3
"""
On-Policy Self-Distillation (TRL-based) Training Script.

Based on "Self-Distillation Enables Continual Learning" - https://arxiv.org/abs/2601.19897
Original implementation: https://github.com/idanshen/Self-Distillation

Key idea: Self-Distillation Fine-Tuning (SDFT) uses demonstration-conditioned model as its own
teacher, generating ON-POLICY training signals that preserve prior capabilities while acquiring
new skills.

How it works:
    1. Teacher sees: prompt + gold answer demonstration → generates response distribution
    2. Student sees: prompt only → learns to match teacher's distribution
    3. Completions are GENERATED during training (on-policy), not read from dataset
    4. KL divergence computed between student and teacher distributions

Example usage:
    python train.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_file_path data/my_dataset.parquet \
        --prompt_key "question" \
        --gold_answer_key "answer" \
        --output_dir checkpoints/opsd_run \
        --learning_rate 2e-5 \
        --num_train_epochs 1
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

import argparse
import logging
import torch
import torch.distributed as dist
from string import Template
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset, load_from_disk

from opsd.trainer import DistilTrainer
from opsd.config import DistilConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_bool(v):
    """Parse boolean string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')


def parse_args():
    parser = argparse.ArgumentParser(description="On-Policy Self-Distillation Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name (defaults to model_name)")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "better-than-sft"), help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name (defaults to output_dir basename)")
    
    # Data arguments
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--use_load_from_disk", type=parse_bool, default=False, help="Use load_from_disk instead of load_dataset")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Key for prompt column")
    parser.add_argument("--gold_answer_key", type=str, default="answer", help="Key for gold answer column")
    parser.add_argument("--teacher_prompt_template", type=str, 
                       default="{prompt}\n\nThis is an example for a response to the question:\n{gold_answer}\n\nNow answer with a response of your own, including the thinking process.",
                       help="Template for teacher prompt")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Distillation arguments
    parser.add_argument("--alpha", type=float, default=0.0, help="KL type: 0=forward, 1=reverse, 0.5=JSD")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty to base model")
    parser.add_argument("--sync_ref_model", type=parse_bool, default=True, help="EMA update teacher")
    parser.add_argument("--ref_model_sync_steps", type=int, default=1, help="Steps between ref model syncs")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="EMA mixup alpha")
    parser.add_argument("--generate_from_teacher", type=parse_bool, default=False, help="Generate from teacher model")
    parser.add_argument("--num_loss_tokens_to_skip", type=int, default=None, help="Tokens to skip at start of completion (default: 0 if force_thinking_prefix is set, else 3)")
    parser.add_argument("--force_thinking_prefix", type=str, default="", help="Prefix appended to prompts after chat template to force thinking (e.g. '<think>\\n')")
    parser.add_argument("--enable_thinking", type=str, default="default",
                        choices=["true", "false", "default"],
                        help="Control Qwen3 thinking mode in chat template during training. "
                             "'true'=enable, 'false'=disable, 'default'=don't pass (tokenizer default)")
    
    # Generation arguments
    parser.add_argument("--num_generations", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    
    # Speculative generation arguments
    parser.add_argument("--speculative_generation", type=parse_bool, default=False, help="Enable speculative KD generation (student proposes, teacher verifies via top-K)")
    parser.add_argument("--speculative_block_size", type=int, default=5, help="Tokens per speculative proposal block (gamma)")
    parser.add_argument("--speculative_acceptance_top_k", type=int, default=25, help="Top-K threshold for teacher acceptance")

    # Chimeric splice generation arguments
    parser.add_argument("--splice_generation", type=parse_bool, default=False,
                        help="Enable chimeric two-pass splice generation (see DistilConfig.splice_generation)")
    parser.add_argument("--splice_starter", type=str, default="student", choices=["student", "teacher"],
                        help="Which role runs Pass 1 of the splice")
    parser.add_argument("--splice_k", type=float, default=0.5,
                        help="Fraction of Pass 1 tokens used as prefix for Pass 2 (in (0, 1))")
    # On-policy demonstration arguments
    parser.add_argument("--gate_mode", type=str, default="none", choices=["none", "wrong_only"],
                        help="'wrong_only': apply the JSD loss only to rollouts with a wrong (and optionally "
                             "different-from-context) final answer; failed rows are regenerated then masked")
    parser.add_argument("--gate_max_regen_rounds", type=int, default=3,
                        help="Fixed full-batch regeneration rounds to replace rollouts failing the gate")
    parser.add_argument("--gate_require_diff_answer", type=parse_bool, default=True,
                        help="Gated rollouts must differ (math-equivalence) from the 'wrong_answer' column")
    parser.add_argument("--gate_gold_answer_key", type=str, default=None,
                        help="Dataset column holding the scalar gold answer used for gate grading")
    parser.add_argument("--gate_wrong_answer_key", type=str, default=None,
                        help="Dataset column holding the in-context wrong answer used for gate grading")
    parser.add_argument("--use_onpolicy_demos", type=parse_bool, default=False,
                       help="Use correct on-policy completions as teacher demos instead of gold answers")
    parser.add_argument("--onpolicy_demo_reward_threshold", type=float, default=1.0,
                       help="Minimum reward to consider a completion correct for on-policy demo")
    parser.add_argument("--strip_thinking_from_demo", type=parse_bool, default=True,
                       help="Strip <think>...</think> from on-policy completion before using as demo")
    parser.add_argument("--demo_reward_function", type=str, default=None,
                       help="Importable path to reward function with signature (prompt_text, completion_text) -> float")

    # Critique-conditioned self-teaching arguments
    parser.add_argument("--use_critique", type=parse_bool, default=False,
                       help="Enable critique-conditioned self-teaching")
    parser.add_argument("--critique_max_prompt_length", type=int, default=8192,
                       help="Max prompt length for critique generation call")
    parser.add_argument("--critique_default_correct", type=str,
                       default="The student's solution is correct. Their reasoning successfully arrives at the answer.",
                       help="Canned critique string for correct rollouts")
    parser.add_argument("--critique_prompt_template", type=str, default=None,
                       help="Template for critique generation (uses default if not set)")
    parser.add_argument("--teacher_critique_template", type=str, default=None,
                       help="Template for teacher prompt with critique (uses default if not set)")

    # vLLM arguments
    parser.add_argument("--use_vllm", type=parse_bool, default=True, help="Use vLLM for generation")
    parser.add_argument("--vllm_mode", type=str, default="colocate", help="vLLM mode: server or colocate")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3, help="vLLM GPU memory utilization")
    parser.add_argument("--vllm_enable_sleep_mode", type=parse_bool, default=True, help="Enable vLLM sleep mode")
    
    # Logging and saving arguments
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--save_strategy", type=str, default="no", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report to (wandb, none)")
    parser.add_argument("--log_completions", type=parse_bool, default=False, help="Log completions")
    
    # Precision and optimization
    parser.add_argument("--bf16", type=parse_bool, default=True, help="Use bf16")
    parser.add_argument("--fp16", type=parse_bool, default=False, help="Use fp16")
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True, help="Use gradient checkpointing")
    
    # FSDP arguments (default to FSDP with qwen config)
    parser.add_argument("--fsdp", type=str, default="full_shard auto_wrap", help="FSDP configuration string")
    parser.add_argument("--fsdp_config", type=str, default="configs/fsdp_config_qwen.json", help="Path to FSDP config file")
    
    return parser.parse_args()


def prepare_distil_dataset(
    dataset: Dataset,
    prompt_key: str,
    gold_answer_key: str,
    teacher_prompt_template: str,
    seed: int = 42,
    gate_gold_answer_key: str = None,
    gate_wrong_answer_key: str = None,
) -> Dataset:
    """
    Prepare dataset for self-distillation training.
    
    Creates 'prompt' (student input) and 'teacher_prompt' (with demonstration) columns.
    """
    # Interpret \n escape sequences from CLI-passed templates (bash single-quoted
    # strings pass literal backslash-n, not real newlines).
    teacher_prompt_template = teacher_prompt_template.replace("\\n", "\n")
    template = Template(teacher_prompt_template.replace("{prompt}", "$prompt").replace("{gold_answer}", "$gold_answer"))
    
    def format_example(example):
        prompt = example.get(prompt_key, "")
        gold_answer = example.get(gold_answer_key, "")
        
        # Handle list-type gold answers (e.g., multiple steps)
        if isinstance(gold_answer, list):
            gold_answer = '\n'.join(str(item) for item in gold_answer)
        
        teacher_prompt_content = template.substitute(prompt=prompt, gold_answer=gold_answer)
        
        out = {
            "prompt": [{"role": "user", "content": prompt}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt_content}],
            "gold_answer": gold_answer,
        }
        # Carry gate-grading columns through the mapping (all original columns are
        # removed, so anything the wrong-only gate needs must be re-emitted here).
        if gate_gold_answer_key:
            out["gate_gold_answer"] = str(example.get(gate_gold_answer_key, ""))
        if gate_wrong_answer_key:
            out["wrong_answer"] = str(example.get(gate_wrong_answer_key, ""))
        return out
    
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def load_training_dataset(args):
    """Load dataset based on file type and configuration."""
    logging.info(f"Loading dataset from {args.train_file_path}")
    
    if args.train_file_path.endswith('.parquet'):
        logging.info("Detected parquet file")
        raw_dataset = load_dataset("parquet", data_files=args.train_file_path, split="train")
    elif args.train_file_path.endswith('.json') or args.train_file_path.endswith('.jsonl'):
        logging.info("Detected JSON file")
        raw_dataset = Dataset.from_json(args.train_file_path)
    elif args.use_load_from_disk:
        logging.info("Using load_from_disk")
        raw_dataset = load_from_disk(args.train_file_path)
        if hasattr(raw_dataset, 'keys') and 'train' in raw_dataset.keys():
            raw_dataset = raw_dataset['train']
    else:
        logging.info("Using load_dataset")
        raw_dataset = load_dataset(args.train_file_path)
        if hasattr(raw_dataset, 'keys') and 'train' in raw_dataset.keys():
            raw_dataset = raw_dataset['train']
    
    return raw_dataset


if __name__ == "__main__":
    args = parse_args()
    
    # Handle escape sequences in force_thinking_prefix
    if args.force_thinking_prefix:
        args.force_thinking_prefix = args.force_thinking_prefix.replace("\\n", "\n")

    # Resolve num_loss_tokens_to_skip default
    if args.num_loss_tokens_to_skip is None:
        args.num_loss_tokens_to_skip = 0 if args.force_thinking_prefix else 3

    # Set wandb project
    os.environ['WANDB_PROJECT'] = args.wandb_project
    
    # Set run name (default to output_dir basename)
    if args.run_name is None:
        args.run_name = os.path.basename(args.output_dir)
    
    # Set tokenizer name if not provided
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name
    
    logging.info(f"Loading model from {args.model_name}")
    
    # Load models
    model_kwargs = {
        "dtype": torch.bfloat16,
    }
    if "gemma4" not in os.path.basename(args.model_name).lower() and "gemma-4" not in os.path.basename(args.model_name).lower():
        model_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load and prepare dataset
    raw_dataset = load_training_dataset(args)
    train_dataset = prepare_distil_dataset(
        raw_dataset,
        prompt_key=args.prompt_key,
        gold_answer_key=args.gold_answer_key,
        teacher_prompt_template=args.teacher_prompt_template,
        seed=args.seed,
        gate_gold_answer_key=args.gate_gold_answer_key,
        gate_wrong_answer_key=args.gate_wrong_answer_key,
    )

    if args.gate_mode != "none":
        assert not args.speculative_generation, "gate_mode is incompatible with speculative_generation"
        assert not args.splice_generation, "gate_mode is incompatible with splice_generation"
        assert not args.generate_from_teacher, "gate_mode is incompatible with generate_from_teacher"
        assert not args.use_critique, "gate_mode is incompatible with use_critique"
        assert args.gate_gold_answer_key, "gate_mode requires --gate_gold_answer_key"
        assert (not args.use_vllm) or (args.vllm_mode == "colocate" and args.vllm_tensor_parallel_size == 1), \
            "gate_mode requires colocated vLLM with tensor_parallel_size=1 (or non-vLLM generation)"
        logging.info(f"Wrong-rollout gate: ENABLED (mode={args.gate_mode}, "
                     f"regen_rounds={args.gate_max_regen_rounds}, "
                     f"require_diff_answer={args.gate_require_diff_answer})")
    
    logging.info(f"Dataset prepared with {len(train_dataset)} examples")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logging.info(f"Sample student prompt: {str(sample['prompt'])[:200]}...")
        logging.info(f"Sample teacher prompt: {str(sample['teacher_prompt'])[:200]}...")
    
    # Build DistilConfig
    config_kwargs = {
        "seed": args.seed,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        
        # Training hyperparameters
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        
        # Distillation settings
        "alpha": args.alpha,
        "beta": args.beta,
        "sync_ref_model": args.sync_ref_model,
        "ref_model_sync_steps": args.ref_model_sync_steps,
        "ref_model_mixup_alpha": args.ref_model_mixup_alpha,
        "generate_from_teacher": args.generate_from_teacher,
        "num_loss_tokens_to_skip": args.num_loss_tokens_to_skip,
        "force_thinking_prefix": args.force_thinking_prefix,
        "enable_thinking": args.enable_thinking,
        
        # Generation settings
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        
        # Speculative generation settings
        "speculative_generation": args.speculative_generation,
        "speculative_block_size": args.speculative_block_size,
        "speculative_acceptance_top_k": args.speculative_acceptance_top_k,

        # Chimeric splice generation settings
        "splice_generation": args.splice_generation,
        "splice_starter": args.splice_starter,
        "splice_k": args.splice_k,

        # On-policy demonstration settings
        "use_onpolicy_demos": args.use_onpolicy_demos,
        "onpolicy_demo_reward_threshold": args.onpolicy_demo_reward_threshold,
        "strip_thinking_from_demo": args.strip_thinking_from_demo,

        # Wrong-rollout gate settings (contrastive OPSD)
        "gate_mode": args.gate_mode,
        "gate_max_regen_rounds": args.gate_max_regen_rounds,
        "gate_require_diff_answer": args.gate_require_diff_answer,

        # Critique-conditioned self-teaching settings
        "use_critique": args.use_critique,
        "critique_max_prompt_length": args.critique_max_prompt_length,
        "critique_default_correct": args.critique_default_correct,

        # vLLM settings
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
        "vllm_importance_sampling_correction": True,
        
        # Logging and saving
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "report_to": args.report_to,
        "log_completions": args.log_completions,
        
        # Precision
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    
    # Add critique templates if provided (handle \n escape sequences from CLI)
    if args.critique_prompt_template is not None:
        config_kwargs["critique_prompt_template"] = args.critique_prompt_template.replace("\\n", "\n")
    if args.teacher_critique_template is not None:
        config_kwargs["teacher_critique_template"] = args.teacher_critique_template.replace("\\n", "\n")

    # Add FSDP config if provided
    if args.fsdp:
        config_kwargs["fsdp"] = args.fsdp
    if args.fsdp_config:
        config_kwargs["fsdp_config"] = args.fsdp_config
    
    config = DistilConfig(**config_kwargs)
    
    # Log key settings
    logging.info("=" * 60)
    logging.info("DistilTrainer Settings:")
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  Output: {args.output_dir}")
    logging.info(f"  Alpha (KL type): {args.alpha} (0=forward, 1=reverse)")
    logging.info(f"  Generate from teacher: {args.generate_from_teacher}")
    logging.info(f"  Sync ref model (EMA): {args.sync_ref_model}")
    logging.info(f"  Max completion length: {args.max_completion_length}")
    logging.info(f"  Use vLLM: {args.use_vllm}")
    logging.info(f"  Batch size: {args.per_device_train_batch_size}")
    logging.info(f"  Gradient accum: {args.gradient_accumulation_steps}")
    logging.info(f"  Learning rate: {args.learning_rate}")
    logging.info(f"  Epochs: {args.num_train_epochs}")
    logging.info(f"  On-policy demos: {args.use_onpolicy_demos}")
    logging.info("=" * 60)
    
    # Create trainer
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Attach reward function and template if on-policy demos or critique are enabled
    needs_reward_fn = args.use_onpolicy_demos or args.use_critique
    if needs_reward_fn:
        if not args.demo_reward_function:
            raise ValueError(
                "--demo_reward_function is required when --use_onpolicy_demos=True or --use_critique=True. "
                "Provide an importable path like 'reward_countdown.score_for_demo'."
            )
        module_path, func_name = args.demo_reward_function.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        trainer.demo_reward_fn = getattr(mod, func_name)
    else:
        trainer.demo_reward_fn = None

    if args.use_onpolicy_demos:
        trainer.teacher_prompt_template = args.teacher_prompt_template.replace("\\n", "\n")
        logging.info(f"  On-policy demos: ENABLED (reward_fn={args.demo_reward_function}, "
                     f"threshold={args.onpolicy_demo_reward_threshold}, "
                     f"strip_thinking={args.strip_thinking_from_demo})")
    else:
        trainer.teacher_prompt_template = None

    if args.use_critique:
        logging.info(f"  Critique-conditioned teaching: ENABLED (reward_fn={args.demo_reward_function}, "
                     f"critique_max_prompt_length={args.critique_max_prompt_length})")

    # Train
    logging.info("Starting on-policy self-distillation training...")
    trainer.train()
    
    # Save
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(output_dir=args.output_dir)
    
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        logging.info(f"Model and tokenizer saved to {args.output_dir}")
    
    trainer.accelerator.wait_for_everyone()
    logging.info(f"Training complete on rank {trainer.accelerator.process_index}")
    
    if dist.is_initialized():
        dist.destroy_process_group()
