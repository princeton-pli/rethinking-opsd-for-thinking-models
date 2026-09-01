# Self-review — OPSD arm launch files (opsd-arms branch, 2026-08-31 overnight)

Scope: rev-3 token-weight patch (applied), `rl/build_opsd_dataset.py` +
built parquets, `rl/arms/arm{A,B,C}.sbatch`, `rl/arms/armD_star_launch.sh`,
`train.py --teacher_model_name`, `slurm/train.sh` passthroughs. Standing
CR-before-launch rule: this file is that review (no subagents per task
instruction). Items marked **[GPU]** could not be verified without a GPU and
are exactly what the mandatory 1h smokes exist to catch.

## 1. Config-key validity (env -> slurm/train.sh -> train.py -> DistilConfig)

Verified by reading the actual parsers, not by pattern-matching:

| arm env | train.sh var | train.py flag | DistilConfig field | checked |
|---|---|---|---|---|
| TOKEN_WEIGHT_MODE | yes (new) | `--token_weight_mode` choices incl. `tail-window`,`uniform` | `token_weight_mode` | `--help` output on della |
| TOKEN_WEIGHT_TAIL_TOKENS | yes (new) | `--token_weight_tail_tokens` | `token_weight_tail_tokens` (new field) | same |
| TOKEN_WEIGHT_EPSILON/SPAN/MID/PRE_SPAN(_TOKENS) | yes (new) | matching flags | matching fields | same |
| TEACHER_MODEL | yes (new) | `--teacher_model_name` | n/a (train.py-level) | CPU load test |
| MODE=qwen3_opsd | slurm/config.sh -> `FSDP_CONFIG=configs/fsdp_config_qwen3_opsd.json` | `--fsdp_config` | — | file exists |
| GATE_* / OPSD_* / LOSS_MAX_COMPLETION_TOKENS / JSD_CHUNK_SIZE / PROMPT_KEY / GOLD_ANSWER_KEY / TEACHER_PROMPT_TEMPLATE | pre-existing (arm1_launch.sh lineage, unchanged) | pre-existing | pre-existing | diffed against arm1_launch.sh |

- `python train.py --help` parses on della's train venv; all new flags print.
- Hard asserts fire correctly by construction: every non-`none` mode requires
  `--loss_max_completion_tokens 0`; arms export `LOSS_MAX_COMPLETION_TOKENS=0`.
  The teacher swap asserts `sync_ref_model=False` (train.sh default False,
  arms also export it) and `generate_from_teacher=False` (default False).
- Doctests: 33/33 pass on della train venv (`python -c "import doctest,
  opsd.trainer as t; ..."`); py_compile clean (local py3.14 + della py3.10).
- `MAX_STEPS` smoke path: train.sh already passes `--max_steps=${MAX_STEPS:--1}`.
- **Not verified [GPU]**: an actual `train.py` start under torchrun/FSDP with
  the new flags. The parser and config accept them; the smoke run is the
  first true execution of the weighted branch on tensors.

## 2. Dataset-field alignment

`data/raw/countdown_opsd_train.parquet` (3,353 rows) / `_heldout` (373 rows),
built on della by `rl/build_opsd_dataset.py` from the 3,726-row pool.

- `prepare_distil_dataset` consumes `PROMPT_KEY=problem`,
  `GOLD_ANSWER_KEY=teacher_context` with `TEACHER_PROMPT_TEMPLATE='{gold_answer}'`
  (pure passthrough — same mechanism Arm 1 used; no train.py data changes).
- Gate asserts require `GATE_GOLD_ANSWER_KEY`/`GATE_WRONG_ANSWER_KEY` to be
  real columns of the RAW dataset: `answer` and `wrong_answer` are present
  (checked on the built parquet).
- Builder asserts per row: parsed `Numbers:[...]`/`Target:` match exactly once
  and parsed target == pool `answer`; question_id unique; held-out split
  disjoint by construction (asserted). Ran clean over all 3,726 rows.
- Rendered context eyeballed (row 0): v4 head (LIST/TARGET), secrecy
  contract, two attempts separated by `---` in pos_first order, probe tail.
  Max teacher_context = 3,344 tokens (real tokenizer) < OPSD_MAX_PROMPT_LENGTH
  4096 (chat-template overhead is ~10 tokens).
- Template choice (documented in the builder header): `v4_pair_content(...,
  episode=False)`. The OPSD teacher scores student rollouts from generation
  start — there is no partial-trace prefix, so the episode variant's
  "continue from where it leaves off" clause would dangle. **Residual
  mismatch, accepted**: the RL'd teacher was trained on the EPISODE variant
  (with thinking prefixes drawn U[0,1/2], including near-zero ones); probe
  evals under the probe variant showed the large gains (0.732->0.893), so the
  probe-context behaviour is certified, but it is not byte-identical to the
  RL training context.

### Gate semantics on Countdown (known, accepted, should be measured)

The trainer gate (`_gate_status`) grades with `extract_answer_math` +
`math_equal(pred, answer)` where `answer` is the TARGET integer. This is
VALUE equality of the boxed expression — it does not check the
each-number-exactly-once constraint, and it may mis-handle `expr = result`
boxed forms (the shape of the 08-27 grader bug, this time in the gate, not
the eval; the eval-side fix in `CountdownTask.grade` does NOT flow into the
gate, which never calls CountdownTask):

- Value-correct but constraint-violating rollouts -> graded "correct" ->
  excluded. Dilution only (fewer wrong rollouts trained on), not
  contamination.
- Value-correct rollouts boxing `expr = result` -> IF `math_equal` fails to
  parse the equation form they are graded "wrong" -> **contamination**
  (correct-answer rollouts distilled under the wrong-only gate). Measured
  base rate of the `=`-form on this task: 16.5% of rollouts.
- Cheap pre-launch mitigation if desired (one line, mirrors the b0de1b4 fix):
  strip at `=` before `math_equal` in `_gate_status`. NOT done — it edits a
  battle-tested trainer function beyond the reviewed patch; needs Sanjeev's
  call. The smoke logs' `gate/live_fraction` plus a `grep` of a few gated
  rollouts will show the actual rate.

## 3. Teacher-loading path (arms A, C)

**Finding that changed the plan: the merged-model route is unsafe for this
adapter, and the task's premise is falsified by measurement.**

- Measured on della (login node, safetensors only): global max|dW| over all
  252 LoRA modules = **3.0e-4** — NOT > 1e-3 (layer-0 modules: 0.5-1.4e-4).
  The lr-2e-5 run moved weights ~15x more than the homeopathic v1 run
  (2e-5 absmax) but still left per-element deltas at or below bf16 ULP for
  typical weight magnitudes.
- Directly measured bf16-merge survival, `bf16(W+dW) - bf16(W)` vs `dW`, on
  3 representative modules: norm survival 0.52-0.80, cosine 0.39-0.62,
  relative error 0.81-0.93. A bf16 merge (or ANY merged dir — train.py casts
  the teacher to bf16 at load) destroys roughly half the adapter and
  replaces it with quantization noise. The HANDOFF's "never evaluate through
  a bf16 merge" rule APPLIES here.
- **Chosen path: load the UNMERGED adapter.**
  `TEACHER_MODEL=rl/checkpoints/grader_grpo_lora_v4tpl/eval_adapter`;
  `AutoModelForCausalLM.from_pretrained(<adapter dir>)` in the train venv
  loads base+adapter via transformers' peft integration — verified on della
  CPU: returns Qwen3ForCausalLM with 504 injected LoRA params, adapter
  active, base pulled from the local path in adapter_config.json (offline
  OK). Unmerged LoRA computes its delta in ACTIVATION space, so bf16 does
  not round it against the base weights elementwise; every v4tpl eval
  (vLLM enable_lora / fp32 forward) ran this way — the probe gains certify
  this exact regime.
- peft version skew: adapter saved by peft 0.18.1, train venv has 0.17.1.
  Load warns and ignores 4 unknown config keys (`alora_invocation_tokens`,
  `arrow_config`, `ensure_weight_tying`, `peft_version`) — all are
  defaults/None in this adapter, so semantics are preserved. Verified the
  warning text; low risk.
- A merge job using the existing tmp_merge machinery WAS submitted
  (13283067, cpu partition) with a hard Stage-0 assert `max|dW| > 1e-3`; per
  the measurement above it is EXPECTED TO FAIL at Stage 0, by design — its
  log (`rl/checkpoints/grader_grpo_lora_v4tpl/tmp_merge/merge_*.out`)
  documents per-module deltas as the durable record. eval_merged is
  deliberately NOT produced.
- **Not verified [GPU]**: FSDP full-shard wrapping of the peft-injected
  teacher (`prepare_fsdp(ref_model)`), and its `summon_full_params` paths.
  The LoRA modules live inside the auto-wrapped decoder layers so this
  should shard normally, but the armA/armC smokes are the first real test.
  If FSDP chokes on the adapter, fallbacks in order: (1) accelerator
  `prepare_model(evaluation_mode=True)` branch, (2) fp32 merged teacher +
  teacher-dtype flag (larger change, memory re-estimate needed).

## 4. Sizing / time limits (documented arithmetic)

Arm-1 measured throughput (sacct + step counts from logs):

| job | steps | wall | H100-h/step | config |
|---|---|---|---|---|
| 12777077 (Arm 1) | 91 | 13:54:50 | 1.22 | gated, gen 16384, loss 4096 |
| 12836814 (Arm 1 matched) | 57 | 8:56:51 | 1.26 | same |
| 12777218 (dense control) | 57 | 2:40:42 | 0.375 | ungated |

Arms A-C: 3,353 rows / batch 64 (8x1x8) = 53 steps x ~1.25 H100-h/step
= **~66 H100-h**, plus full-window loss overhead (unmeasured; JSD/logit
forwards over up to 16k instead of 4k loss tokens, generation still
dominates) -> estimate **66-75 H100-h**, 8.3-9.4h wall. Time limit
14:00:00 = 49% margin over the high estimate (standing >=30% rule).
Countdown-vs-math unknowns absorbed by the margin: trace-length
distribution (median ~10.2k tokens, close to math's ~10.9k) and gate regen
rate (pool problems average ~50% solve rate, comparable regen load to
Arm 1's initial live fraction 0.42-0.63).

Arm D*: SFT, ~53 steps/epoch of short sequences — minutes/epoch on 4 GPUs.
Compute-matching to 60-80 H100-h would be hundreds of overfitting epochs;
STaR controls are data-matched, not compute-matched. Default 3 epochs,
1h limit (>10x margin). This is a deliberate, documented deviation from the
60-80 H100-h target.

Total if all four launch after smokes: ~200-230 H100-h + 4 smoke-hours
(x8 GPUs for A-C smokes = 24 H100-h) + eval battery (~50-100, as Arm 1).

## 5. Smoke protocol (mandatory, before each arm)

    sbatch --time=01:00:00 --job-name=opsd_arm<X>_smoke \
           --export=ALL,SMOKE=1 rl/arms/arm<X>.sbatch

3 optimizer steps, full-trace loss, nvidia-smi 60s sampling to
`logs/smoke_mem-<job>.csv`, per-GPU peak printed at exit. Smoke uses
`DATASET_NAME=..._smoke` because train.py SAVES A CHECKPOINT even after 3
steps and slurm/train.sh skips training when the checkpoint dir has
config.json — without the suffix the real run would silently no-op (the
standing DATASET_NAME gotcha).

Post-smoke checklist before the real launch:
1. Peak memory.used <= ~70 GB/GPU (estimate ~53 GB + vLLM colocate share;
   80 GB cards). OOM -> raise JSD_CHUNK_SIZE granularity is already 1024;
   next lever is micro-batch (already 1) -> escalate to Sanjeev.
2. Log line `Token-weighted distillation: ENABLED (mode=...)` present with
   the right mode; A/C also `Teacher swap: FROZEN teacher loaded from ...`.
3. `gate/live_fraction` present and sane (0.3-0.95); `weights/no_span_fraction`
   ABSENT (only boxed modes log it).
4. Step wall-time x 53 fits the 14h limit with >=30% to spare.
5. (A/C) no FSDP errors mentioning `lora_A/lora_B` params.

## 6. Other things that could waste a launch

- **Checkpoint identity**: train.sh now appends `__tw_<mode>__tchr_<teacher>`
  to checkpoint names; A/B/C cannot collide with each other, with Arm 1, or
  with their own smokes. Verified by tracing the name construction.
- **Partition**: all arm files carry `#SBATCH --partition=pli-c` explicitly
  (the 2026-08-24 default-partition bite).
- **Absolute cd**: arm files `cd /scratch/gpfs/ARORA/arora/opsd` rather than
  trusting SLURM_SUBMIT_DIR.
- **W&B**: disabled via env pair (the no-tty abort, jobs 12610253/54).
- **Relative TEACHER_MODEL path** resolves against the repo root because of
  the absolute cd; from_pretrained accepts it (CPU-verified with the same
  relative path).
- **armD cross-repo seams**: tokenization uses della-post-training's venv +
  adapter (`prompt_response`), checkpoint redirected into this repo's
  checkpoints/. The launcher prints a decoded tokenized example 0 — eyeball
  it for an unexpected empty `<think>` block in the assistant turn before
  sbatching. **Not verified**: I did not run the tokenize step (writes to
  data/sft/; harmless, but I kept the overnight run read-only outside
  data/raw and rl/). DRY_RUN=1 supported.
- **x_plus is think-stripped** (armD): SFT teaches no-think outputs; the
  control is outcome-supervision-matched, not channel-matched. If Sanjeev
  wants a full-trace STaR control it needs a re-harvest keeping `<think>`
  (GPU, ~2-3h at k=8 on the 3.7k problems, or reuse of existing harvest raw
  files which were NOT retained).
- **Eval battery not auto-chained**: submit after training with the
  arm1_launch.sh pattern (`--dependency=afterok:<jobid>`, shards with
  `SKIP_GENERATE_MERGE=true`, then `./merge_evals.sh <ckpt>`). Kept out of
  the arm files to keep the smoke->launch flow single-command; the eval
  set for the damage check is the Arm-1 math battery and needs Sanjeev's
  §6.3-seam decision (Countdown-trained arms make it a transfer eval).
- **della sync**: rl/ scripts, patched train.py/opsd/*, and slurm/train.sh
  must be byte-identical on della before launch (repo there is file-sync,
  not git). Synced as of this review; re-verify with a checksum pass if
  anything is edited locally afterwards.

## 7. Open items for Sanjeev (decision, not execution)

1. Gate `=`-form fix in `_gate_status` (see §2): accept the ~16% wrong-only
   contamination risk, or approve the one-line strip-at-`=` trainer edit?
2. armD epochs (default 3) and its think-stripped nature — acceptable as the
   STaR control, or re-harvest full traces first?
3. Eval battery for Countdown-trained arms: keep the Arm-1 math battery as a
   transfer/damage eval (proposal §6.3 seam), and/or add a Countdown eval?
4. TOKEN_WEIGHT_EPSILON=0.001 and TAIL=4096 defaults (proposal open q. 5).
5. The v4 episode-vs-probe teacher-context mismatch (§2) — accepted here;
   flag if he wants the episode clause instead.
