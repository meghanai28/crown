# CROWN

Crazy long term goal: Can an agent learn the geometry of specialization well enough to recursively navigate weight space and spawn task-specific experts, rather than relying on blind random perturbation?


**Do task-specific subspaces exist in weight space, and can a model
learn to find them?**

The long-term goal is a navigator: something that reads a task and
proposes a direction to move the base model's weights in, instead of
guessing randomly. That only makes sense if directions have structure
worth predicting. Stage 1 checks whether they do. Everything else is
gated on it.

## Stage 1: the smoke test

`python stage1_density.py` produces one table:

```
                     math          code        KL
   isotropic      ?/10 ( ?%)    ?/10 ( ?%)   0.050
  math-delta      ?/10 ( ?%)    ?/10 ( ?%)   0.050
 coder-delta      ?/10 ( ?%)    ?/10 ( ?%)   0.050
```

Each row is a source of random directions. Each cell is how often a
random direction from that source made the model better at that family.

- **Diagonal beats off-diagonal** → task-specific subspaces carry
  task-specific information. Stage 2 has something to learn.
- **All rows equal** → subspace geometry carries nothing. The navigator
  idea loses its foundation.
- **Both deltas win everywhere** → the prior is doing something generic,
  not something task-specific. Also a negative.

### This is not task arithmetic

The obvious objection: *of course moving along a math delta improves
math.* That would be true, and it would prove nothing — it is model
merging, and it is well known.

That is not what happens here. **The delta's values are never used.**
Only the subspace its top singular vectors span is used. Directions are
then drawn *at random inside that subspace* — as likely to point away
from the math checkpoint as toward it, with an expected inner product of
zero against the delta itself.

So the question is not "does the delta help". It is: **is the
low-dimensional region a specialist was built in, a region where other
useful models also live?** That is a claim about geometry, not about the
particular checkpoint, and it is the claim a navigator would need to be
true.

This also dodges the reason the naive version is ill-posed. Two RL runs
from the same base produce near-orthogonal updates: they disagree about
*what* to write. A model trained to regress onto those weights would
average incompatible solutions. But such runs do appear to agree about
*where* they write. So this project learns where, never what.

### Why each arm gets its own scale

Say you want to compare "push north" against "push east" fairly. You
have to push equally hard in both directions, or you learn nothing about
direction — only about force.

Weight-change size is the wrong measure of force. Measured here, the
same `||dW||/||W|| = 0.08` gave:

| direction | next-token KL |
| --- | --- |
| spread isotropically | 0.185 |
| packed into a rank-32 subspace | **0.543** |

Same weight change, **2.9× the effect on the model's output**. Squeezing
a fixed amount of change into fewer directions hits the model harder —
like the same total force through a needle instead of a palm.

So an earlier run where the subspace arm looked worse was not evidence
about subspaces at all; that arm was simply being hit three times
harder. The script now calibrates each arm's scale to reach the same KL
before sampling, so the arms differ in direction and not in force.

### What is deliberately kept out of the score

The score is `log P(correct answer) - log P(a wrong answer)` with the
`<answer>` tag teacher-forced. Two artifacts are excluded by
construction, both of them things that already fooled this repository
once:

- **Formatting.** The first experiment's apparent 18-point gain came
  from a direction that made the model emit `<answer>` before running
  out of tokens. Forcing the tag removes that lever.
- **Confidence.** A direction that only sharpens the output
  distribution raises the probability of *any* short string, which looks
  identical to knowing more answers. Subtracting a matched wrong answer
  cancels it. This is not decorative — without it, an early arm looked
  strongly positive on math and most of the gain vanished once the
  distractor term was subtracted.

### Honest limitation: this is not a RandOpt reproduction

The perturbations here are **low-rank**, not full-parameter. At rank 16
they span 0.62% of the targeted weight space:

| rank | free parameters | % of weight space |
| --- | --- | --- |
| 16 (default) | 40.4M | 0.62% |
| 64 | 161.5M | 2.5% |
| 256 | 645.9M | 9.9% |
| full | 6.53B | 100% |

Full-parameter noise would need a 12.2 GB dense delta and dense matmuls
against a quantized base. `RANK` is the knob; rank 256 costs about
2.4 GB of factors.

**What this does and does not invalidate.** It means nothing here
reproduces RandOpt, and no claim of the form "random search fails"
should be made from these runs — that would need full-parameter noise
and populations in the thousands. It does *not* invalidate stage 1,
because every arm is rank-constrained in the same way. The isotropic arm
is really "a random rank-16 subspace" and the delta arm is "a specific
rank-32 subspace", so the comparison is subspace-versus-subspace, which
is exactly the question. If anything that is the cleaner contrast.

## Stage 2: the part that actually learns

Stage 1 only establishes that the signal exists. It does not produce a
navigator, and a delta you already own is not useful for a task you do
not have a checkpoint for. The point of stage 1 is to justify stage 2.

If the diagonal shows up, the data to train on is already a by-product
of stage 1: for many tasks, many sampled directions, each with a
measured reward. That gives triples of

```
(task evidence, direction coefficients, reward)
```

A small network trained on that predicts *which coefficients are worth
trying for a new task*. Two properties make it the right shape:

- **Trained on reward, not reconstruction.** Regressing onto saved
  checkpoints is the approach that fails, because it averages
  incompatible solutions. Ranking directions by measured reward has no
  such problem.
- **Cheap to store.** A direction is a seed plus a scale plus a handful
  of coefficients, so the archive does not grow with model size.

The test for stage 2 is a held-out *task*, not a held-out example: does
a predicted direction beat random directions from the same subspace at
equal budget? Only then is it a navigator rather than a lookup table.

## What the first experiment established

`select_specialists.py` searched 16 random directions and kept the best
by dev-set accuracy. It reported `code dev 50.0% -> 68.8%` and
`test 56.2% -> 56.2%` — a large gain on the set used to pick the winner
and zero transfer. Four reasons that null says nothing about the
hypothesis:

- **Not significant.** McNemar exact on the dev flips gives p = 0.25
  before correcting for the search, and p = 0.99 after. The average
  random direction already scored 8.69/16 against the base's 8.0 with no
  selection at all, because churn on a failing item is free upside — a
  wrong answer can only stay wrong or become right.
- **The gains were artifacts.** All three flipped items were format
  failures: one scorer bug (a regex capturing the colon in "the result
  is: [0, 2, 4, 6]", since fixed) and two answers truncated by the
  196-token cap before the tag.
- **The probe was tiny.** Rank-4 on 4 of 28 layers is 0.027% of the
  weight space.
- **The holdout was not held out.** It is category-paired with dev — the
  same 16 operations with different arguments.

Random sampling survives as the control arm. It is no longer the method.

## Running it

```
python scoring.py              # scorer unit tests, no model needed
python verify_perturbation.py  # perturbation is real and reverts exactly
python subspace.py             # bases are orthonormal and actually restrict
python stage1_density.py       # the smoke test
```

Base model is `Qwen2.5-7B-Instruct-4bit`. The two deltas come from
`Qwen2.5-Math-7B-Instruct` and `Qwen2.5-Coder-7B-Instruct`, which share
its architecture *and* its quantization recipe (4-bit, group 64), so
subtracting them cancels most of the quantization error and leaves the
specialization. 7B is the largest size where this works: `Qwen2.5-Math`
exists only at 7B and 72B, so moving to 14B would cost the math arm and
with it the negative control.

Three checkpoints, about 13 GB on disk. Roughly 25 minutes for the
default 10 samples per arm. Each delta is loaded once to build 196
bases, then released.

Before sampling, each delta arm reports how concentrated its subspace
is. The Math-7B delta's rank-32 subspace holds **14.2%** of the delta's
energy where an unstructured source would give **0.0175%** — an ~800×
concentration, so the delta really is low-rank. **If that number ever
comes back near the unstructured figure, the prior is noise and the
run means nothing.**

With 10 samples the standard error on a density near 50% is ~16 points.
Only a large gap means anything; raise `SAMPLES_PER_ARM` for a real
result.

## Layout

```
model.py                # frozen checkpoint, generation, answer_logprob
tasks.py                # dev and held-out task suites
scoring.py              # format / value / semantic decomposition
perturb.py              # low-rank directions, isotropic or subspace-restricted
subspace.py             # checkpoint deltas -> cached randomized bases
verify_perturbation.py  # numerical and restoration check
stage1_density.py       # the smoke test
select_specialists.py   # the original accuracy search, kept for reference
```

## Perturbation semantics

For every targeted matrix, `||dW||_F / ||W||_F = |scale|`. The same seed
at different scales is the same direction at different distances;
different seeds are different directions. `subspace_weight` is the
fraction of the update's energy drawn inside the subspace, mixed by
concatenating along the rank axis, since
`[a1 | a2] @ [[b1], [b2]] == a1 @ b1 + a2 @ b2`.

The sampler was restructured to support mixing, so a given seed no
longer produces the direction it did before that change. Seed-for-seed
comparisons against older results are invalid.

Do not read unchanged greedy text as proof of an inactive perturbation.
Check the logit deltas from `verify_perturbation.py` first.

## Limits

- One 4-bit 7B checkpoint on one machine. Anything here may be a fact
  about this model at this quantization.
- The deltas are not pure task-training directions. Math-7B and Coder-7B
  branch from Qwen2.5-7B rather than from its instruct variant, so each
  difference mixes specialized pretraining with a different instruct run.
- Teacher-forced answer log probability is not chain-of-thought
  accuracy. It measures whether the value is available directly, which
  is right for a density probe and wrong for reporting task performance.
- 16 tasks per family is enough to compare arms that share seeds, and
  not enough to report an absolute capability number.
