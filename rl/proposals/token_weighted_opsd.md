# Token-weighted contrastive-OPSD — design note

Status: PROPOSAL, 2026-08-30. Nothing applied, nothing committed, nothing run.
Companion patch: `rl/proposals/token_weighted_opsd.patch` (apply-ready,
verified with `git apply --check` against `contrastive-opsd` HEAD; default-off,
no-op unless `--token_weight_mode boxed_hybrid` is passed).
Prior context: HANDOFF_2026-08-27.md (Arm 1), HANDOFF_2026-08-30.md (sparse-signal
result), `rl/eval_sparse_signal.py`, `rl/sparse_signal_base.jsonl`.

## 1. Motivation

The 2026-08-30 sparse-signal measurement localized the contrastive signal Arm 1
failed to distill. At the wrong trace's own `\boxed{` slot, pair context moves
the frozen teacher toward the correct expression by a **pair-specific +0.925
nats (29σ, positive on 90.9% of 922 wrong rollouts)** — bare −1.43 → pair
−0.50. Meanwhile the pair perturbs conditionals broadly along the trace
(0.4–0.9 nats/token of pair-conditioned *style* KL, early-mid-trace prose
heavy), so a uniform distillation loss spends ~99% of its gradient on style and
buries the slot signal **~100:1** (whole-trace margin ≈ 0; the handoff's
answer-token dilution estimate is ~1500x). Arm 1 distilled exactly that
style — and got −13.4 pts and a −32% length collapse. The hypothesis here:
keep everything from Arm 1 (frozen teacher, unlabeled pair context, wrong-only
gate, no teacher training) and change ONLY the loss weighting — concentrate it
on the answer region of wrong rollouts, where the measured mechanism lives.

## 2. Where the weights inject (blast radius)

The whole loss reduction is one line:

- `opsd/trainer.py:2651` —
  `per_seq_loss = (per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)`

`loss_completion_mask` is already the composition of attention mask, the
`num_loss_tokens_to_skip` prefix skip (`trainer.py:2577–2584`), gate
zero-masking (`trainer.py:2364–2366`), and the loss-window truncation
(`trainer.py:2380–2385`). A per-token float weight tensor multiplies in at
exactly this point. The weights are built in
`_generate_and_score_completions` — after gate regeneration
(`trainer.py:2134–2168`, so they see the final `completion_ids_list`) — and
ride the buffered-inputs dict through `shuffle_sequence_dict` /
`split_tensor_dict` (`trainer.py:990–995`) the same way
`importance_sampling_ratio` (also shape B×T) already does. No change to
generation, the gate, the teacher, tokenization, or the JSD itself.

**The one structural tension (stated plainly):** the injection is clean, but
*exploiting* it is not free. Arm 1 computed the loss on the **first 4096**
tokens (`LOSS_MAX_COMPLETION_TOKENS=4096`, `arm1_launch.sh:49`) while the
boxed span sits at the trace **end** (median rollout ~10.9k tokens; at a 4096
cap 95% never reach `\boxed{`). Token-weighting the answer region therefore
requires the loss window to cover the full trace. Two ways:

1. **Full window** (recommended): run with `--loss_max_completion_tokens 0`.
   Zero new alignment code. Memory: the vocab-wide fp32 log-softmax is ~9 GiB
   per tensor at T=16384 (config.py's own arithmetic) vs ~2.3 GiB at T=4096;
   student (autograd-retained) + teacher (no-grad, alive through the chunked
   JSD) ⇒ estimated +13–14 GiB over Arm 1's measured 38.9/81.6 GB rung-A
   peak ⇒ ~53 GB, inside an 80 GB H100. `jsd_chunk_size=1024` already
   checkpoint-chunks the divergence block. **This is an estimate — a 1-hour
   rung-A smoke is mandatory before the run.** Compute cost of longer loss
   forwards is real but small next to 16k-token generation + gate regen.
2. **Suffix window** via `logits_to_keep < completion length`: memory-optimal
   but touches mask/importance-sampling alignment in two battle-tested
   functions (~20 lines of slicing surgery). Not worth it unless the smoke
   OOMs. Deferred.

The patch hard-asserts `loss_max_completion_tokens == 0` when weighting is
enabled, so the silently-null configuration (span truncated out of the loss,
every row ~zero weight) is impossible to launch.

## 3. Weighting options

Effective weight-mass shares at the Arm-1 median trace (10,900 tokens,
boxed span ≈ 15 tokens, pre-span window k = 256):

| profile | span share | pre-window | off-window | notes |
|---|---|---|---|---|
| uniform (Arm 1) | 0.14% | — | 99.9% | the measured ~100–1500x burial |
| (a) hard window: 1 on span, ε=0.001 elsewhere | 58% | — | 42% | ~15 loss-relevant tokens/rollout → high gradient variance |
| (b) soft ramp (exp toward end, τ=1000 / τ=100) | 1.5% / 14% | — | rest | concentrates only as τ → span size, i.e. degenerates to (a); loads late-trace *prose*, which the KL profile says is style |
| (c) hybrid: ε=0.001 + 1.0 on span + 0.05 × 256 pre-span | 39% | 33% | 28% | **recommended** |
| (c) with ε=0.01 | 11% | 10% | 79% | ε dominates again — ε must stay ≤ ~0.001 |

**Recommendation: (c) hybrid**, defaults ε=0.001, span=1.0, pre=0.05, k=256.
Reasons: the measured signal is at the slot, but the pre-span region (the
final-answer statement after `</think>`) plausibly carries the decision tokens
feeding it; the window hedges ±1-token locator error at span boundaries; and
~270 weighted tokens per rollout instead of ~15 keeps per-step gradient noise
sane. (a) is the natural ablation and is reachable by setting
`--token_weight_pre_span_tokens 0`. (b) is dominated: it is a worse-
parameterized (c), and end-loading prose weight is the direction the KL
localization says is inert.

All four numbers are flags (`token_weight_epsilon / _span / _pre_span /
_pre_span_tokens`), so the profile is a launch-time choice, not a code change.

## 4. Design decisions

**Normalization.** Per-rollout **weight-normalized mean**:
`per_seq_loss = Σ(loss·w·mask) / Σ(w·mask)`. Equivalent to normalizing each
rollout's weights to sum to 1, so every rollout contributes at the same scale
regardless of trace length or profile, and batches stay comparable across
steps and to Arm 1 (which used the same structure with w≡1). The gate's
live-row averaging on top is preserved unchanged.

**Wrong-only gate.** Untouched. Weights multiply the same
`loss_completion_mask` the gate zeroes; the live-row denominator now uses the
combined weights, so rows dropped for *either* reason (gate-dead or no boxed
span) do not dilute the step. Right rollouts stay excluded (recommend keeping
this matched to Arm 1 for attribution; see open questions).

**Rollouts with no boxed span.** These exist among gate-live rows:
`extract_answer_math` (evaluation/utils.py:189–221) grades via "the answer
is"/last-number fallbacks, so a rollout can be gradeably-wrong without a
literal `\boxed{`. Decision: **drop** (all-zero weight, same semantics as a
gate failure), counted in a new `weights/no_span_fraction` metric. Falling
back to uniform is rejected: it would silently reintroduce full-trace style
distillation on an unmonitored subset — the exact failure mode being removed.
If the metric comes back high (>10–15%), tighten the gate to require a boxed
answer rather than soften the weights.

**Length-collapse interaction (Arm 1's −32%).** Two channels considered.
(i) The collapse mechanism Arm 1 exhibited — imitating the context-conditioned
teacher's early-commitment style across the whole trace (the probe showed
context roughly halves responses) — is attacked head-on: off-window gradient
drops ~100x. Expectation is mitigation, not aggravation. (ii) "Weighting the
end rewards early termination" — there is no reward here and the loss acts on
given rollouts, so no direct incentive; the indirect route is on-policy drift
(shorter student → shorter rollouts → different loss support). Mitigations,
all pre-registered as hard stops rather than post-hoc reads:
`completions/mean_length`, `mean_terminated_length`, `clipped_ratio`, and
`kl_approx` are already logged every step; kill the run if mean terminated
length falls below 90% of the step-1–5 baseline for 10 consecutive steps
(Arm 1's collapse was visible well inside 28 steps). Note `kl_approx` stays
full-trace deliberately — it now monitors style drift exactly where the loss
no longer acts. Eval adds the budget-curve crossing check (4–8k vs ≥16k),
which was Arm 1's signature.

**Span locator.** Pure function `locate_boxed_span(ids, decode)` (module
level in trainer.py): finds the last `\boxed{` and its matching brace in
decoded text, then maps char→token by binary search over prefix-decode
lengths — tokenizer-agnostic, no assumption about how `\boxed{` splits into
tokens. Unclosed box (truncated mid-box) clips to trace end; nested braces
match the outer close; multiple boxes take the last. Five doctest cases in
the docstring (all verified passing). Cost: ~14 prefix decodes per rollout
per generation, negligible next to generation itself.

**Known approximations, left alone deliberately (minimal patch):** the vLLM
importance-sampling correction (`trainer.py:2644`) still averages the ratio
over the *unweighted* mask — it is a per-sequence scalar correction and
reweighting it would couple two mechanisms in one change. The
`num_loss_tokens_to_skip=3` prefix skip composes trivially (span is nowhere
near token 3).

## 5. What the patch contains

Three files, +190/−2 lines, default-off (with `token_weight_mode="none"` every
executed line is byte-equivalent to today):

- `opsd/config.py` — five `DistilConfig` fields after the gate block.
- `opsd/trainer.py` — `locate_boxed_span` (pure, doctested); weight
  construction after gate regen + loss-window truncation; passthrough in the
  output dict next to `importance_sampling_ratio`; weighted branch at the
  `per_seq_loss` reduction; one new metric.
- `train.py` — five CLI flags, config passthrough, fail-loud asserts
  (`loss_max_completion_tokens == 0`; no speculative/splice/teacher-gen).

Not in the patch (launch-time follow-ups, keep the proposal no-op):
`slurm/train.sh` needs a `TOKEN_WEIGHT_MODE` env passthrough mirroring
`GATE_MODE` (slurm/train.sh:69,178), and the checkpoint-name suffix should
encode the mode so a rerun can't silently reuse the Arm-1 checkpoint
(the standing `DATASET_NAME` gotcha). Per standing rule, a CR pass on the
applied patch precedes any launch.

## 6. Eval plan (same instruments + damage check)

1. **Training monitors** (every step, with the pre-registered stop rule):
   lengths, clipped ratio, `kl_approx`, `gate/live_fraction`,
   `weights/no_span_fraction`.
2. **Margin/probe instruments**, unchanged from the signal run:
   `rl/eval_logprob_margin.py` + `rl/eval_sparse_signal.py` on the trained
   student vs base — did the bare-context slot margin move toward correct
   (the distillation target), and did whole-trace KL to base stay below
   Arm 1's (style suppression working)?
3. **Damage check**: AIME24/25 + HMMT25 avg@16 at 38,912, same battery as
   Arm 1 — reference points: base 0.599, dense-gold control 0.542 (−5.7),
   Arm 1 matched 0.465 (−13.4). Plus the budget-curve crossing.
   Read: success = slot-margin transfer with damage strictly inside the
   control's −5.7; stretch = ≥ base.

## 7. Estimated run cost

Arm-1 scale assumed: 3.6k–5.8k problems, 57–91 steps at effective batch 64
(v3 pooled parquet = 4,823 rows ≈ 75 steps). On pli-c (allocation, not
dollars):

| item | estimate |
|---|---|
| rung-A memory smoke (mandatory) | ~8 H100-h (1h × 8) |
| training, 8×H100, ~12–18h (Arm 1 asked 20h; +10–30% for full-window loss forwards) | ~100–150 H100-h |
| eval battery, 3 sets × 8 shards | ~50–100 H100-h |
| margin/probe instruments | ~2–4 H100-h |
| **total** | **~160–260 H100-h** |

## 8. Open questions for Sanjeev

1. **Weight profile**: hybrid (c) as primary with hard-window (a) as the
   one ablation — or (a) only for a cleaner single-mechanism read?
2. **ε**: 0.001 (28% off-window mass) vs 0 (pure window)? ε>0 hedges locator
   misses and keeps a whisper of full-trace distillation — but full-trace
   distillation is precisely what hurt in Arm 1; a case for ε=0 exists.
3. **Right rollouts stay excluded?** Recommend yes (matched to Arm 1, isolates
   the weighting delta). The alternative — small-weight right rollouts as a
   stabilizer — is a second mechanism and muddies attribution.
4. **No-span rows**: drop (implemented) vs tightening the gate to require a
   boxed answer. Decide after seeing `weights/no_span_fraction`.
5. **Loss window**: accept the full-window memory estimate pending the rung-A
   smoke, or pre-authorize the suffix-window surgery as fallback if it OOMs?
6. **Stop rule**: is 90%-of-baseline terminated length for 10 consecutive
   steps the right pre-registered kill threshold?
7. **Eval plan** (§6) sign-off, and whether `gate_require_diff_answer` stays
   False as in Arm 1.
