# Spec: RL the teacher into a better grader (hard Countdown)

Status: draft for review, 2026-08-27. Author: Claude, from §6 of
HANDOFF_2026-08-27 plus decisions taken in chat (Sanjeev: LoRA first,
spec-then-launch, skaur dirs readable).

## 1. Objective

Train the OPSD teacher (Qwen3-4B) to grade reliably in the regime OPSD
actually uses it: a single forward pass over student tokens, conditioned on an
unlabeled (x⁺, x⁻) pair. The training task is NOT verdict emission:

> given (problem, unlabeled correct+wrong responses, partial thinking trace)
> → produce (completed thinking trace, right answer).

Output stays think-then-answer: no verdict register to leak into ordinary
solving, no negation phrasing to invert, and it trains exactly the quantity
OPSD consumes — P(next | ctx, prefix) — rather than hoping adjudication shows
up in the conditionals.

## 2. Evidence base (all 2026-08-27, honest grader)

* Math probe: adjudication 54.4% nothink / 62.8% think on clean pairs.
  Verification ≈ generation → contrastive-OPSD starved. Arm 1 −13.4 matched.
* Countdown v2 probe (158 pairs, label noise 0.6%): picks the right candidate
  when copying **75.4% nothink / 90.0% think**; none/nothink accuracy 0.133 vs
  unlabeled/nothink 0.690. The verification–generation gap exists here, and
  most of it is already visible without thinking.
* Neglabel backfires on Countdown too (picks x⁻ 28.6% vs 24.1% nothink;
  26.6% vs 9.8% think). Negative labels stay out of all designs.
* 3–4 number Countdown cannot supply training data: 75.2% of problems solved
  8/8, usable-pair rate 7.9% (158/2000). Hence the hard generator.

## 3. Hard-Countdown generator (`pilot/gen_hard_countdown.py`)

Solvable by construction; **harden the task, don't filter it** (filtering
conditions on model failure → biased sub-distribution).

* Sample n ∈ {5, 6} numbers uniformly from [1, 99] (with replacement across
  draws, multiset within an instance allowed — matches source data marginals).
* Build a uniformly random binary expression tree over ALL n numbers, ops
  {+, −, ×, ÷}. Rejection constraints, mirroring classic Countdown and the
  existing data: every intermediate value a positive integer; final value
  (= target) in [10, 100] (source data: targets 10–100, numbers 1–99).
* Reject degenerate constructions: any intermediate identical to a leaf it
  combines with via ×1, ÷1, or a−b=0 shortcut (they reduce effective n).
* Reject instances solvable by any ± combination of the numbers (≤64 evals;
  CR 2026-08-27 measured 72% of unfiltered witnesses were pure signed sums —
  a ≤32-combo search that would recreate the too-easy regime). Escape hatch:
  `--allow_signed_sum` if the gate shows the filtered set is too hard.
* Dedup on (sorted number multiset, target). Store the witness expression in
  the parquet for auditability; the witness NEVER enters any prompt.
* Prompt template: same as source data but "each number **exactly once**" —
  fixes the at-most-once/exactly-once prompt–grader mismatch (7.7% of old x⁻
  were subset solutions).
* Output schema: identical to `countdown_15k.parquet` normalized columns so
  `countdown_pairs.py` and `CountdownTask.grade` work unchanged.

Difficulty is not assumed — it is measured by the harvest (§4). If 5-number
instances still land >50% solved-8/8, shift the mix toward 6 (knob, not
redesign).

## 4. Gate: harvest + probe before any RL

Same pipeline as v2, ~2.5 h on one H100, no training:

1. Generate 3,000 instances (buffer over 2,000 to survive dedup/rejection).
2. k=8 harvest with the fixed grader → `pilot/countdown_pairs_hard.parquet`.
3. 3×2 adjudication probe → grade with `grade_countdown_probe.py`.

Proceed to RL only if:

* **usable-pair rate ≥ 25%** (v2 baseline 7.9%; below 25% the RL data-diet is
  too thin and the mix shifts to 6 numbers / wider n before spending more), and
* **adjudication when copying ≥ 65% nothink** on hard pairs (the gap must
  survive hardening; if verification collapses along with generation, the
  (model, task) pair is wrong and we stop here — that is the probe doing its
  job, ~1 GPU-hour instead of a failed training run).

## 5. RL design

* **Trainer**: verl in `della-post-training` (cross-repo). opsd exports a
  self-contained parquet of episodes; a thin config + reward file lives with
  the trainer. verl is NOT vendored into opsd.
* **Model**: Qwen3-4B, **LoRA** (r=32, α=64, all linear) — teacher-with/
  without-adapter becomes a free ablation and the frozen base is always
  available as reference. Full-FT only if LoRA shows signal but saturates.
* **Episode**: prompt = problem + unlabeled-pair context (verbatim Arm-1
  template, randomized order) + `<think>` prefix of a harvested trace.
  Prefix length uniform in [0, ½·trace] (matches OPSD's own loss window,
  first 4,096 of ~10,900 tokens ≈ 38%). Prefixes come from BOTH right and
  wrong rollouts of the SAME problem — steering away from a bad prefix and
  staying on a good one are both in-distribution.
* **Reward**: outcome-only. Last `\boxed{}` graded by the deterministic
  checker (eval + exactly-once multiset). The trace is never graded.
* **Known blind spot, measured not rewarded**: ignore-the-prefix-and-resolve
  earns full reward while teaching nothing about steering. Measurement, per
  eval batch: (a) fraction of completions whose expression reuses an
  intermediate value computed in the prefix; (b) completion-length vs
  from-scratch-length distribution; (c) 50-sample LLM-judge spot check
  (g-flash, local, cached). If restart dominates, that is a finding about the
  reward, reported not patched silently.
* **KL budget**: KL(trainee ‖ frozen base) measured on the generative
  distribution over held-out OPSD-style rollouts (the distribution OPSD will
  score), logged every eval; hard-stop threshold set after step-0 calibration.
* **Length is the leading indicator** (Arm 1: −32% within 28 steps, visible
  long before accuracy moved). Response-length curves on both eval sides from
  step 0.

## 6. Evaluation (two-sided from the start)

* **Grader side**: 3×2 adjudication probe cells before/after; completion
  accuracy from held-out prefixes (right-prefix and wrong-prefix separately).
* **Generative side**: AIME24/25 + HMMT25 avg@16 at 4k–38,912 budgets +
  response length + fork rate — the Arm-1 harness unchanged.
* **End-to-end (separate follow-up run, own approval)**: plug the RL'd
  teacher into contrastive OPSD on hard Countdown; success = beats both the
  frozen-teacher contrastive arm and the dense-gold control.

## 7. Budget

| Stage | Est. | Cap |
|---|---|---|
| Generator + tests | laptop, $0 | — |
| Hard harvest + probe | ~3 H100-h | 1 job, 10 h limit |
| LoRA RL (first signal run) | ~50–100 steps, 8–16 H100-h | 24 H100-h |
| g-flash judge spot checks | <$1 | $2 |

GPU spend beyond the gate in §4 needs no further sign-off (approved in chat
2026-08-27); the end-to-end OPSD rerun in §6 does.

## 8. Open questions (for Sanjeev, non-blocking defaults chosen)

1. Prefix source: harvested rollouts only, or also the witness-expression
   trace rendered as a synthetic solution? Default: rollouts only (witness
   stays quarantined; synthetic traces are a distribution shift).
2. Should wrong-prefix episodes overweight prefixes whose wrong rollout
   reached a wrong FINAL answer (vs died mid-trace)? Default: natural mix.
3. GRPO vs PPO in verl. Default: GRPO (group size 8, matches k and needs no
   value model). The paper-side repo (skaur) also trains with GRPO.
4. n ∈ {5,6} mix. Default: 50/50, rebalanced once by the §4 measurement.
