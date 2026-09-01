#!/bin/bash
#SBATCH --job-name=grader_grpo
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:4
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/rl/slurm-%j.out
#
# Grader-RL run (SPEC §5/§7): Qwen3-4B + LoRA on prefix-continuation
# episodes. verl 0.7.0 from della-post-training's rlvr venv.
# Budget: 4 H100 x 6h = 24 H100-h, the spec cap for the signal run.
#
# NOTE ON NAMING: the objective is REINFORCE WITH A GROUP-MEAN BASELINE, not
# GRPO proper. verl's `adv_estimator=grpo` is only the config key for the
# group-relative advantage; std scaling (norm_adv_by_std_in_grpo) and length
# normalization (loss_agg_mode) are both OFF, leaving A = R - mean(R_group)
# with token-sum aggregation, plus a 1e-3 KL to the frozen model. The
# "grpo" in the filename and experiment name is legacy.
#
# CRITICAL 0.7.0 facts (verified against installed source + smoke test):
#  * No sync rollout; generation goes through the AgentLoop, which applies
#    the chat template -- the IDENTITY template below is what makes our
#    mid-<think> plain-string prompts reach vLLM verbatim. Do not remove.
#  * data.truncation is inert; filter_overlong_prompts (custom, plain-string
#    aware) is the real guard.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source /scratch/gpfs/ARORA/arora/della-post-training/envs/rlvr/.venv/bin/activate || exit 1

EPISODES=${EPISODES:-rl/episodes_balanced_v4tpl}
[ -f "$EPISODES/train.parquet" ] || { echo "episodes missing: $EPISODES"; exit 1; }
python rl/reward_countdown.py || { echo "reward regression cases FAILED"; exit 1; }

export WANDB_MODE="offline"
# openai_harmony (pulled in by this venv's verl) downloads tiktoken vocabs at
# init; compute nodes have no internet. Same fix as della-post-training's
# slurm_rlvr.sh: point it at the pre-downloaded cache.
export TIKTOKEN_ENCODINGS_BASE=/scratch/gpfs/ARORA/arora/della-post-training/envs/harmony_encodings
MODEL=/scratch/gpfs/ARORA/skaur/models/Qwen3-4B
OUT=${OUT:-/scratch/gpfs/ARORA/arora/opsd/rl/checkpoints/grader_grpo_lora_v4tpl}
TPL="{% for m in messages %}{{ m['content'] }}{% endfor %}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$EPISODES/train.parquet" \
    data.val_files="$EPISODES/test.parquet" \
    data.prompt_key=prompt \
    data.custom_cls.path=/scratch/gpfs/ARORA/arora/opsd/rl/prefix_dataset.py \
    data.custom_cls.name=PrefixContinuationDataset \
    "+data.apply_chat_template_kwargs.chat_template=\"$TPL\"" \
    data.train_batch_size=64 \
    data.max_prompt_length=12288 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUT" \
    trainer.project_name=grader_rl \
    trainer.experiment_name=grader_grpo_lora_v4tpl \
    trainer.balance_batch=False \
    custom_reward_function.path=/scratch/gpfs/ARORA/arora/opsd/rl/reward_countdown.py \
    custom_reward_function.name=compute_score

echo "GRADER GRPO RUN DONE"
