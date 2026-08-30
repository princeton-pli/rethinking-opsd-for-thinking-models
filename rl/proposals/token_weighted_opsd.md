# Token-weighted contrastive-OPSD — design note

Status: PROPOSAL, rev 2, 2026-08-30. Nothing applied, nothing committed,
nothing run. Rev 2 replaces the recommended profile: **numeric-skeleton**
(boxed span + every intermediate computation-result token), superseding the
rev-1 boxed-hybrid recommendation after the mid-trace slip measurement.
Companion patch: `rl/proposals/token_weighted_opsd.patch` (apply-ready,
verified with `git apply --check` against `contrastive-opsd` HEAD (9a908b9);
default-off, no-op unless `--token_weight_mode` is passed). Both modes ship
in the patch; `boxed_hybrid` is retained as an ablation arm.
Prior context: HANDOFF_2026-08-27.md (Arm 1), HANDOFF_2026-08-30.md
(sparse-signal result + LATE ADDITIONs), `rl/eval_sparse_signal.py`,
`rl/eval_midtrace_slip.py`, results `rl/sparse_signal_base*.jsonl`,
`rl/midtrace_slip_base.jsonl` (della).

## 1. Motivation

Three measurements since Arm 1, all on the frozen teacher (Qwen3-4B, fp32):

1. **Answer-slot signal** (`rl/eval_sparse_signal.py`): at the wrong trace's
   own `\boxed{` slot, pair context moves the teacher toward the correct
   expression by a pair-specific **+0.925 nats (29σ, positive on 90.9% of
   922 wrong rollouts)** under the v1 template, **+0.842 nats (27σ, 91%)**
   under v2 — the mechanism is phrasing-robust. Meanwhile the pair perturbs
   conditionals broadly along the trace (0.4–0.9 nats/token of
   pair-conditioned *style* KL), so a uniform distillation loss buries the
   slot signal ~100:1. Arm 1 distilled exactly that style — −13.4 pts and a
   −32% length collapse.
2. **Mid-trace signal** (`rl/eval_midtrace_slip.py`, new): at the FIRST FALSE
   arithmetic statement in a wrong trace (`a op b = c` with false `c`,
   scored at the result slot given the identical prefix), the pair boosts the
   arithmetically TRUE result by **+1.389 nats (31.0σ, positive on 92.5% of
   600 slip sites)**. The effect on the FALSE (written) result is +0.05
   (1.7σ) — the pair steers *toward truth* rather than merely suppressing the
   slip. So contrastive signal is not slot-only: it exists mid-trace and is
   **localized on intermediate computation-result tokens**. This kills the
   main objection to slot-only weighting: a slot-only distillate ≈ "fix the
   final answer", which plain STaR/RFT on x⁺ already does; the mid-trace
   component is what only distillation can transmit.
3. **Prose contamination**: pair-conditioned *prose* references the pair
   heavily under v2. The training template is now v4 (no-reference,
   commit dd20677), which suppresses references generatively — but weighting
   only NUMERIC tokens sidesteps prose contamination entirely, independent of
   how well any template suppresses it. A number cannot smuggle "as attempt
   2 says".

The hypothesis is unchanged from rev 1: keep everything from Arm 1 (frozen
teacher, unlabeled pair context, wrong-only gate, no teacher training) and
change ONLY the loss weighting — concentrate it where the measured mechanism
lives, now including the mid-trace computation results.

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
   functions (~20 lines of slicing surgery) — and is now also *wrong for
   numeric-skeleton*, whose weighted tokens are spread over the whole trace.
   Dead option under the new profile except as a boxed-hybrid fallback.

The patch hard-asserts `loss_max_completion_tokens == 0` when weighting is
enabled, so the silently-null configuration (span truncated out of the loss,
every row ~zero weight) is impossible to launch.

## 3. Weighting options

Rev 2 grounds the weight-mass table in **measured** trace statistics rather
than rev 1's assumed ones: over the first 400 balanced Countdown prefix
rollouts (`pilot/prefix_rollouts_balanced.jsonl`, Qwen3-4B tokenizer),
median trace = 10,165 tokens, median `a op b = c` matches = 238/trace,
median result tokens = 452/trace, median boxed span = 25 tokens, 89% of
rollouts contain `\boxed{`. (Caveat: this is the Countdown distribution —
arithmetic-dense by construction. Math-arm density is unmeasured; see open
question 4.)

Effective weight-mass shares at those medians:

| profile | span | mid (results) | pre-window | ε off-span | notes |
|---|---|---|---|---|---|
| uniform (Arm 1) | 0.25% | — | — | 99.7% | the measured ~100:1 burial |
| (a) hard window: 1 on span, ε=0.001 | 72% | — | — | 28% | ~25 loss tokens/rollout → high gradient variance; distills "fix the answer", which RFT gets free |
| (c) rev-1 hybrid: span 1.0 + 0.05×256 pre-span + ε | 52% | — | 27% | 21% | pre-window is *prose* — the contamination channel; retained as ablation |
| **(d) numeric-skeleton: span 1.0 + w_mid=0.2 on results + ε** | **20%** | **72%** | — | **8%** | **recommended**; every weighted token is numeric |
| (d) with w_mid=0.05 | 44% | 39% | — | 17% | span-dominant variant |
| (d) with w_mid=0.5 | 10% | 87% | — | 4% | mid dominates; probably too far |

**Recommendation: (d) numeric-skeleton**, defaults ε=0.001, span=1.0,
w_mid=0.2. Reasons: the mid-trace signal is bigger per-token than the slot
signal (+1.39 vs +0.84 nats) and sits exactly on these tokens; ~477 weighted
tokens/rollout (vs 25 for (a)) keeps per-step gradient noise sane; and the
numeric-only support is contamination-proof by construction — the rev-1
pre-span prose window is deleted from the recommended profile for exactly
that reason. At w_mid=0.2 the mid mass dominates (72%); whether that is the
right split against the span is open question 1. All numbers are flags
(`token_weight_epsilon / _span / _mid`), so the profile is a launch-time
choice, not a code change.

**Design choice, alternative flagged — weight ALL results, not only false
ones.** Locating truth at training time is free (each `a op b = c` is
self-contained arithmetic; `eval_midtrace_slip.py::first_slip` already does
it), so a false-only variant is implementable. We weight all results anyway:
true results also carry signal (reinforcing correct-arithmetic conditionals
under pair context), and false-only weighting would make the weighted-token
count collapse on mostly-correct traces. The false-only (or
false-upweighted) variant is the natural ablation if (d) underperforms; it
is NOT implemented in the patch.

## 4. Design decisions

**Normalization.** Per-rollout **weight-normalized mean**, unchanged from
rev 1: `per_seq_loss = Σ(loss·w·mask) / Σ(w·mask)`. Every rollout
contributes at the same scale regardless of trace length or weight profile;
the gate's live-row averaging on top is preserved unchanged.

**Result locator.** Pure function `locate_result_spans(ids, decode)`
(module level in trainer.py, next to `locate_boxed_span`): regex
`(\d+)\s*([-+*/×÷x])\s*(\d+)\s*=\s*(-?\d+)` — the same EQ_RE family as
`rl/eval_midtrace_slip.py` — and the **group-4 span** (the stated result
`c`) is mapped char→token by the same binary search over prefix-decode
lengths as the boxed locator. Doctests cover multiple equations, mixed
true/false results, unicode operators (× ÷), signed results, and the
no-arithmetic case. Verified against the real Qwen3-4B tokenizer on a
synthetic trace (result spans decode back to the result numerals; boundary
slop is ≤1 token, e.g. a result token absorbing an adjacent space).

**Overlap rule.** `numeric_skeleton_weights` (pure, doctested) applies
ε → result spans at w_mid → boxed span at w_span, in that order, so **the
boxed-span weight wins wherever a result span overlaps the box** (Countdown
answers are literally `a op b = target` inside `\boxed{}`). Doctested and
checked against the real tokenizer.

**Wrong-only gate.** Untouched. Weights multiply the same
`loss_completion_mask` the gate zeroes; the live-row denominator uses the
combined weights, so rows dropped for *either* reason do not dilute the
step. Right rollouts stay excluded (open question 6).

**Rollouts with no boxed span.** Same rev-1 semantics in both modes:
**drop** (all-zero weight), counted in `weights/no_span_fraction`. Falling
back to mid-only weighting was considered for numeric-skeleton (the mid
spans exist without a box) and rejected for now: no-box rollouts are
truncated/degenerate traces and a silent regime change on an unmonitored
subset is the Arm-1 failure shape. Measured floor: 11% of the balanced
rollouts have no box, so expect `no_span_fraction` ≈ 0.1.

**Length-collapse interaction (Arm 1's −32%).** As in rev 1: off-span
gradient drops ~100–1000x, attacking the style-imitation channel head-on;
no direct incentive toward early termination (no reward, loss on given
rollouts). Pre-registered hard stop unchanged: kill the run if mean
terminated length falls below 90% of the step-1–5 baseline for 10
consecutive steps. `kl_approx` stays full-trace deliberately — it monitors
style drift exactly where the loss no longer acts.

**Locator cost (new, honest).** The boxed locator does ~14 prefix decodes
per rollout. The result locator does ~2×238×14 ≈ 6.7k prefix decodes per
rollout at Countdown density, each O(prefix length) — an estimated seconds
per rollout, unprofiled at training scale. It runs once per generation
batch, next to 16k-token generation + gate regen, so it should disappear in
the noise; if profiling says otherwise, the drop-in fix is a shared
prefix-length cache across the ~500 boundaries of a rollout (pure-function
refactor, no semantics change). Flagged in the self-review.

**What the JSD target at a mid-trace slot actually is
(expectation-setting).** The teacher's distribution at a slip site still
prefers the written false result by ~10 nats (margin(true−false):
bare −11.08 → pair −9.74). Distilling at those slots transfers a ~1.4-nat
*shift* toward truth, not a flip. The bet is that many small conditional
shifts on computation results compound over a trace; the instrument to
check post-training is the same `eval_midtrace_slip.py` margin on the
student.

**Known approximations, left alone deliberately (minimal patch):** the vLLM
importance-sampling correction (`trainer.py:2644`) still averages the ratio
over the *unweighted* mask — a per-sequence scalar correction; reweighting
it would couple two mechanisms in one change. The
`num_loss_tokens_to_skip=3` prefix skip composes trivially.

## 5. What the patch contains

Three files, +333/−2 lines, default-off (with `token_weight_mode="none"`
every executed line is byte-equivalent to today):

- `opsd/config.py` — six `DistilConfig` fields after the gate block
  (rev 1's five + `token_weight_mid`, default 0.2); mode help documents both
  profiles and both measurements.
- `opsd/trainer.py` — `locate_boxed_span`, `locate_result_spans`,
  `numeric_skeleton_weights` (all pure, 25 doctest examples total, all
  passing on della's train venv via
  `python -c "import doctest, opsd.trainer as t; print(doctest.testmod(t))"`
  — note `python -m doctest` on the file fails on the module's relative
  import); weight construction after gate regen supporting both modes;
  passthrough next to `importance_sampling_ratio`; weighted branch at the
  `per_seq_loss` reduction; `weights/no_span_fraction` metric.
- `train.py` — six CLI flags (`--token_weight_mode` now with a
  `numeric-skeleton` choice, `--token_weight_mid`), config passthrough,
  fail-loud asserts (`loss_max_completion_tokens == 0`; no
  speculative/splice/teacher-gen).

Verification actually run for rev 2: `git apply --check` against HEAD
9a908b9; round-trip (HEAD + patch ≡ edited copies, byte-identical);
`py_compile` on all three modified copies (local py3.14 + della venv
py3.10); all 25 doctests on della; real-tokenizer end-to-end check of the
three locator/weight functions.

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
2. **Margin/probe instruments**: `rl/eval_logprob_margin.py` +
   `rl/eval_sparse_signal.py` on the trained student vs base (did the bare-
   context slot margin move toward correct, and did whole-trace KL to base
   stay below Arm 1's?), **plus rev 2's new instrument**:
   `rl/eval_midtrace_slip.py` on the student — did the bare-context TRUE-
   result logprob at slip sites rise? That is the direct read on whether the
   mid-trace conditionals transferred.
3. **Damage check**: AIME24/25 + HMMT25 avg@16 at 38,912, same battery as
   Arm 1 — reference points: base 0.599, dense-gold control 0.542 (−5.7),
   Arm 1 matched 0.465 (−13.4). Plus the budget-curve crossing.
   Read: success = slot AND slip-site margin transfer with damage strictly
   inside the control's −5.7; stretch = ≥ base.

## 7. Estimated run cost

Unchanged from rev 1 (the locator adds CPU seconds per batch, not GPU
hours): ~160–260 H100-h on pli-c — mandatory 1-h rung-A memory smoke ×8,
training ~100–150, eval battery ~50–100, instruments ~2–4.

## 8. Open questions for Sanjeev

1. **w_mid value**: 0.2 makes mid-trace results 72% of the weight mass and
   the boxed span 20% (measured medians). Right split? 0.05–0.1 makes it
   span-dominant. The per-token signal argues for mid-heavy (+1.39 vs +0.84
   nats); gradient-variance and "the slot is the outcome" argue for
   span-heavy. Default 0.2 unless overridden.
2. **True vs false results only**: patch weights ALL `a op b = c` results
   (see §3 design choice). The false-only variant is free to locate and is
   the natural ablation. Primary = all-results, false-only as follow-up if
   (d) underperforms — agreed?
3. **Interaction with the v4 template**: both measurements were made under
   v1/v2 pair phrasing; training now uses v4 (no-reference). The slot signal
   was phrasing-robust v1→v2 (+0.925→+0.842), so v4 is expected to preserve
   it, but neither the slot nor the slip increment has been re-measured
   under v4. Re-run both probes under v4 (cheap, existing scripts) before
   the training run, or accept the v2 numbers as sufficient?
4. **Distribution mismatch in the density stats**: 238 equations/trace is
   Countdown; math traces (the Arm-1 damage-check distribution) are
   arithmetic-sparser, which shifts (d) toward (a) automatically via the
   normalization. Measure EQ_RE density on Arm-1-style math rollouts first,
   or accept the profile as-is (it degrades gracefully)?
5. **ε**: 0.001 (8% off-span mass under (d)) vs 0 (pure skeleton)? ε>0
   hedges locator misses; full-trace distillation is what hurt in Arm 1, and
   under (d) ε mass is already 3x smaller than rev 1's.
6. **Right rollouts stay excluded?** Recommend yes (matched to Arm 1,
   isolates the weighting delta).
7. **No-span rows**: drop (implemented) vs mid-only fallback vs tightening
   the gate to require a boxed answer. Expect `no_span_fraction` ≈ 0.1 on
   Countdown-style data; revisit if higher.
8. **Loss window**: accept the full-window memory estimate pending the
   rung-A smoke (suffix-window fallback is now incompatible with (d)).
9. **Stop rule**: 90%-of-baseline terminated length for 10 consecutive
   steps, carried from rev 1.
10. **Eval plan** (§6) sign-off, incl. the new slip-site instrument, and
    whether `gate_require_diff_answer` stays False as in Arm 1.

## 9. Self-review (rev 2, things I am not sure about)

- **Which training distribution this arm actually targets.** The damage
  check (§6.3) is the Arm-1 math battery, but every rev-2 measurement and
  density statistic is Countdown. If the run trains on Countdown pairs, §6.3
  is a transfer eval, not a damage check, and the reference points (0.599 /
  0.542 / 0.465) are from a different data regime. I carried the Arm-1 eval
  plan forward as instructed but this seam should be resolved explicitly
  (open questions 3–4 touch it; the run config decides it).
- **Locator cost is estimated, not profiled** (~6.7k prefix decodes/rollout
  at Countdown density). Mitigation is known and cheap (shared prefix-length
  cache) but I did not pre-build it, to keep the patch minimal. If the
  rung-A smoke shows the generation loop slowing, this is the first suspect.
- **The regex is deliberately narrow.** Integer-only `a op b = c`; it misses
  decimals, multi-term chains (`2+3+4 = 9` matches only `3+4 = 9`… actually
  it matches `3\s*\+\s*4 = 9` — the left operand of the match is the last
  number, so chains are partially captured with a wrong-looking `a`), `=`
  written as "equals", and LaTeX inline math. For weight *placement* only
  the group-4 span matters, so wrong-looking left operands are harmless, but
  coverage on math-style traces (fractions, symbolic steps) will be much
  lower than on Countdown.
- **±1-token boundary slop**: with the real tokenizer a result span can
  absorb an adjacent space/punctuation token (observed: `' -3'`), and the
  boxed span can absorb a leading space / trailing period token. Harmless at
  these weight scales; noted so nobody is surprised reading weighted-token
  dumps.
- **The mid-trace measurement is at FIRST-slip sites only** (that is what
  `eval_midtrace_slip.py` scores); the patch weights ALL result tokens,
  including hundreds of true ones per trace. The +1.39-nat number therefore
  does not directly certify the average weighted token — the all-results
  choice rests on the §3 argument, not on a measurement of true-result
  sites. A cheap extension of the eval script could measure the true-result
  sites too.
- **Slip prevalence**: the fraction of wrong rollouts containing any false
  arithmetic statement is printed by the eval script's run log, which I did
  not have; `rl/midtrace_slip_base.jsonl` holds the 600 slip rows but not
  the denominator. If prevalence is low, mid-trace mass lands mostly on true
  results, which sharpens open question 2.
- **Naming inconsistency, kept per spec**: mode string `numeric-skeleton`
  (hyphen) next to `boxed_hybrid` (underscore). Cosmetic; flag if it should
  be normalized before the CR pass.
- Verified myself rather than trusting rev 1's claims: rev 1's patch still
  applies at HEAD 9a908b9; its doctest instruction (`python -m doctest
  opsd/trainer.py`) does not actually work against the package-relative
  import — rev 2's docstrings and §5 state the working invocation.
