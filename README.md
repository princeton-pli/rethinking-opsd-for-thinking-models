# Rethinking On-Policy Self-Distillation for Thinking Models

This repository contains the code for our paper
[Rethinking On-Policy Self-Distillation for Thinking Models](https://arxiv.org/abs/2607.05184).

On-policy self-distillation (OPSD) trains a student against a teacher that is
a *frozen copy of the same checkpoint*, with the teacher's context augmented
by privileged information (a gold demonstration, or just the final answer).
The student samples its own rollouts; a token-level generalized JSD pulls the
student toward the privileged teacher on those rollouts. We find that this
degrades thinking models at long rollout budgets — this repo contains the
training, long-budget evaluation, and rollout-budget-sweep code behind those
experiments.

## Quick Links

- [Requirements](#requirements)
- [Data](#data)
- [Experiments](#experiments)
- [Evaluation](#evaluation)
- [Bugs or questions?](#bugs-or-questions)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Requirements

Python 3.10 with torch 2.9.0, transformers 4.57.1, trl 0.24.0, vllm 0.12.0,
and accelerate 1.11.0 (the exact versions used for the paper's runs). Two
ways to install:

```bash
# Option A: uv (recommended; installs the locked environment into envs/train/.venv)
bash envs/setup.sh
source envs/train/.venv/bin/activate

# Option B: pip
pip install -r requirements.txt
```

Download the models (defaults to `./models`; override `MODELS_DIR`):

```bash
for m in Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B \
         Qwen/Qwen3-4B-Thinking-2507 Qwen/Qwen3-4B-Instruct-2507 \
         allenai/Olmo-3-7B-Think allenai/Olmo-3-7B-Instruct; do
    hf download "$m" --local-dir "models/$(basename $m)"
done
```

Experiments run on SLURM (paper setup: 8×H100 nodes for training, single-GPU
array jobs for evaluation). Cluster-specific settings (partition, module
loads, NCCL flags) go in `slurm/cluster_env.sh`:

```bash
cp slurm/cluster_env.sh.example slurm/cluster_env.sh   # then edit
```

## Data

Training data is built from Hugging Face (nothing is redistributed here):

```bash
python data_tools/build_openthoughts_30k.py        # clean OpenThoughts math (29,439 rows)
python data_tools/make_openthoughts_15k_subset.py  # first 15,000 rows = main training subset
python data_tools/build_countdown.py               # Countdown 15k train / 500 held-out (seed 42)
python data_tools/build_eval_sets.py               # AIME24/25 + HMMT25 parquets
```

For OpenThoughts, the student prompt is the `problem` column and the teacher's
privileged context is the full `solution` (dense) or the boxed `Answer`
(sparse). For Countdown, the prompt is `datapoint_input_text` and the
privileged context is `response_suffix` (dense) or `response_answer_only`
(sparse). AIME24/25 and HMMT25 are fetched from public Hugging Face datasets
(see `data_tools/build_eval_sets.py`); you can also drop your own
`problem`/`answer` parquets into `data/eval/`.

## Experiments

Every run from the paper is a named row in
[submit_experiment.sh](submit_experiment.sh), which submits training plus the
chained evaluations:

```bash
./submit_experiment.sh list                    # all 25 experiment names
./submit_experiment.sh qwen3_4b_ot15k_dense    # a single experiment
DRY_RUN=1 ./submit_experiment.sh all           # print the sbatch commands only
```

Experiment groups:

| Command | Experiments |
|---|---|
| `./submit_experiment.sh openthoughts` | OPSD with dense (full gold demonstration) context on OpenThoughts-15k, for the five thinking models |
| `./submit_experiment.sh sparse` | the sparse (final-answer-only) counterparts, for the dense-vs-sparse budget comparisons |
| `./submit_experiment.sh countdown` | OPSD on Countdown-15k for paired instruct/thinking models |
| `./submit_experiment.sh countdown_sparse` | Countdown sparse counterparts |
| `./submit_experiment.sh nothink` | thinking vs. thinking-disabled OPSD training on OpenThoughts-30k |
| `./submit_experiment.sh concise` | conciseness-instruction control (teacher gets no gold context) |

Shared training setup: generalized JSD with α=0.5 on on-policy student
rollouts (colocated vLLM; sampling T=1.0, top-p 1.0, top-k 50), frozen
self-teacher, 1 epoch, effective batch 64, completion cap 4,096 tokens,
cosine schedule with warmup ratio 0.03, bf16, FSDP, seed 42. Dense runs use max
prompt length 25,000 and LR 5e-6 (Qwen3-8B: 2e-6); sparse runs use 8,192.
Checkpoints land in `checkpoints/opsd__m_<model>__d_<data>__...` with the
full setting string in the name.

## Evaluation

Each experiment automatically evaluates on AIME24, AIME25, and HMMT25 (and
the held-out Countdown split for Countdown runs): 16 samples/problem at a
38,912-token generation cap, sharded over 8 single-GPU jobs, T=0.6 / top-p
0.95 (Qwen3-4B-Instruct-2507: T=0.7 / top-p 0.8). Results land in
`<checkpoint>/eval/<task>_38912/`: the merged generations JSONL and a
`metrics.json` with `accuracy` (= avg@16) and unbiased `pass@k`.

**Baselines**: evaluate the untrained models the same way by pointing
`CHECKPOINT_PATH` at the model directory:

```bash
for eval_name in aime24_38912 aime25_38912 hmmt25_38912; do
    sbatch --export=ALL,EVAL_SET_NAME=${eval_name},EVAL_MAX_TOKENS=38912,NUM_SHARDS=8,CHECKPOINT_PATH=models/Qwen3-4B,TEMP=0.6,N_SAMPLES=16,TOP_P=0.95,EVAL_BATCH_SIZE=8 \
        slurm/generate_with_retry.sh
done
```

**Budget curves**: compute metrics at various rollout budgets from an
experiment's generations:

```bash
python run_budget_sweep.py \
    --file_path checkpoints/<run>/eval/aime24_38912/<merged>.jsonl \
    --tokenizer_path checkpoints/<run>
# writes metrics_budget{4096,8192,16384,32768}.json next to the input,
# including mean/median response-token stats for the length panels
```

Rough cost: training is ~5–12 h on one 8×H100 node per run (4×H100 for most
sparse runs); each evaluation task is 8 single-GPU shards of up to ~4 h at
the 38,912-token cap (the retry wrapper resumes partial shards). Budget
sweeps are CPU-only.

## Repository layout

```
train.py, opsd/                OPSD trainer (frozen privileged teacher, JSD on
                               on-policy rollouts, colocated vLLM generation)
configs/                       FSDP wrap policies (Qwen3, OLMo-3)
submit_experiment.sh           all paper runs, one name each
slurm/                         job templates (train / generate / retry / eval)
run_generate.py                sharded, resumable vLLM generation
run_eval.py                    accuracy + unbiased pass@k -> metrics.json
run_budget_sweep.py            metrics at various rollout budgets
evaluation/                    eval tasks (AIME24/25, HMMT25, Countdown) + math grading
data_tools/                    dataset builders
envs/                          uv-locked environment (requirements.txt mirrors it)
```

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to
email Simran (`skaur 'at' princeton 'dot' edu`). If you encounter any
problems when using the code, or want to report a bug, you can open an
issue. Please try to specify the problem with details so we can give more
effective help!

## Citation

```bibtex
@article{kaur2026rethinking,
  title={Rethinking On-Policy Self-Distillation for Thinking Models},
  author={Kaur, Simran and Ri, Narutatsu and He, Yinghui and Fowl, Liam and Arora, Sanjeev},
  journal={arXiv preprint arXiv:2607.05184},
  year={2026}
}
```

## Acknowledgements

- The OPSD trainer (`opsd/`, `train.py`) is adapted from
  [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation),
  the reference implementation of *"Self-Distillation Enables Continual
  Learning"* ([arXiv:2601.19897](https://arxiv.org/abs/2601.19897)), and
  builds on [TRL](https://github.com/huggingface/trl) trainer internals.
- `evaluation/grader.py` and `evaluation/utils.py` (math answer extraction and
  SymPy equivalence) are adapted from
  [TianHongZXY/RLVR-Decomposed](https://github.com/TianHongZXY/RLVR-Decomposed).
- Generation and evaluation run on [vLLM](https://github.com/vllm-project/vllm).
- Data: [Ashkchamp/Openthoughts_math_filtered_30K](https://huggingface.co/datasets/Ashkchamp/Openthoughts_math_filtered_30K)
  (derived from [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)),
  [jasonrqh/Countdown-CoT-20k](https://huggingface.co/datasets/jasonrqh/Countdown-CoT-20k),
  and public AIME 2024 / AIME 2025 / HMMT February 2025 competition problems
  via Hugging Face mirrors.

This code is released under the Apache-2.0 license (see [LICENSE](LICENSE)).
