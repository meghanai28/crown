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
- **Both deltas win everywhere** → structure helps but task identity does
  not. With only math and code this outcome is *ambiguous*, not a
  negative — see below.

## What stage 1 found (100 samples per arm)

```
                  math        code      scale      KL
   isotropic    41/100      18/100     0.0493   0.0499
  math-delta    50/100      37/100     0.0186   0.0532
 coder-delta    52/100      34/100     0.0127   0.0485
```

Two findings, one of them not the one being tested.

**Structure beats isotropic, on code.** Both deltas roughly double the
isotropic rate on code (37 and 34 against 18; z = 3.01, p = 0.003 and
z = 2.58, p = 0.010). On math the gap is not significant (50 and 52
against 41; p = 0.20 and p = 0.12). So *where* you perturb matters.

**Task identity does not.** The two deltas are statistically
indistinguishable from each other: p = 0.78 on math, p = 0.66 on code.
There is no diagonal. Worse, the specificity is faintly inverted —
against a baseline asymmetry of `math - code = +23` for isotropic,
math-delta gives +13 and coder-delta gives -18. Relative to chance
neither delta favours its own domain. The apparent "math-delta helps
math" is only that math is easier to improve than code.

**Why that is ambiguous rather than negative.** Math and code are
unusually entangled for an LLM: symbolic manipulation, stepwise
reasoning, exact-output behaviour, overlapping post-training data. Two
stories fit equally well — a shared *formal-reasoning* region, or a
generic *high-plasticity* region that post-training carves out
regardless of task. Behaviour cannot separate them.

**Geometry can, and does.** `python subspace_overlap.py` measures the
principal angles between the two delta subspaces directly, no sampling
needed:

| matrices | side | overlap | chance | ratio |
| --- | --- | --- | --- | --- |
| attention | left | 0.1161 | 0.0357 | 3.3x |
| attention | right | 0.0550 | 0.0089 | 6.2x |
| MLP | left | 0.0608 | 0.0041 | **14.8x** |
| MLP | right | 0.1183 | 0.0065 | **18.2x** |

**The Math and Coder deltas overlap about 10.6x above chance**, and much
more strongly in the MLP than in attention. A random-subspace control
reproduces the analytic chance floor almost exactly (0.0359 against
0.0357; 0.0036 against 0.0041), so the ratio is real.

They are not the *same* subspace — a mean squared cosine of 0.06 to 0.12
is far from 1.0 — but they are nowhere near independent. So the two
deltas behave identically in part because they are substantially writing
to the same place. That is a partially shared region plus task-specific
remainder, and this experiment cannot yet see the remainder.

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

### The density metric was measuring the wrong thing

Counting "how often did a direction improve the score" rewards a
distribution squeezed toward zero, because a zero-centred distribution
crosses zero about half the time by construction. That is exactly what
the delta arms produce. At matched KL their margin changes have roughly
half the spread of isotropic:

| arm | std dev (math) | std dev (code) |
| --- | --- | --- |
| isotropic | 0.340 | 0.254 |
| math-delta | 0.189 | 0.124 |
| coder-delta | 0.156 | 0.106 |

Ask instead how often a direction produced a *large* gain, which is what
a search actually wants, and the ranking inverts:

| threshold | isotropic | math-delta | coder-delta |
| --- | --- | --- | --- |
| math, Δ > +0.1 | **36** | 30 | 32 |
| math, Δ > +0.2 | **28** | 15 | 11 |
| math, Δ > +0.3 | **23** | 5 | 4 |
| code, Δ > +0.2 | **3** | 1 | 1 |

Isotropic finds large math gains four to six times more often. The delta
subspaces do not find better checkpoints; they find *smaller* changes in
both directions. They are safer, not better, and the headline density
was rewarding safety.

So as a **sampling prior for search, the delta subspace fails.** If you
want to find a better model, aim anywhere and keep the winners.

### But the geometry is real, and it is not generic adaptation

`subspace_overlap.py`, with `instruct-control` = Qwen2.5-7B minus
Qwen2.5-7B-Instruct, a post-training delta with no task specialization:

```
                      math      coder   instruct-control
   math                 -       12.9x        4.0x
   coder              12.9x       -          3.4x
   instruct-control    4.0x      3.4x         -
```

Math and coder are **12.9x** aligned with each other but only 3.4-4.0x
aligned with generic post-training. And projecting the instruct subspace
out barely touches it:

```
   math vs coder:  raw 12.9x  ->  after 11.9x   (96.0% of energy kept)
```

The sharing survives removing the generic direction almost completely.
**That is task-family geometry, not a generic safe-to-edit region.**

### What the two results mean together

The subspace is real, specific to the math-and-code family, and
separable from generic adaptation. But it is also a *low-sensitivity*
region: at equal next-token KL, perturbing inside it moves task margins
much less than perturbing anywhere. Both facts at once say the subspace
is not useful as a place to sample randomly.

That does not kill the navigator; it relocates it. The subspace's value
is as a **coordinate system**, not as a sampling prior. Stage 2 should
learn *which coefficients within the basis* are worth using, rather than
drawing uniformly inside it and hoping. Random sampling wastes the
structure this experiment just demonstrated exists.

## Stage 1b: distant domains

Math and code are too close together to tell a formal-reasoning region
from a universal one. The fix is domains with genuinely different
computational demands, not just different benchmark names:

| domain | demand |
| --- | --- |
| math | symbolic, multi-step reasoning |
| code | algorithmic, formal execution |
| translation | linguistic mapping, almost no arithmetic |
| biomedical QA | factual knowledge retrieval |
| summarization | compression, discourse generation |

Then read the full matrix. A block structure where math and code cluster
together and translation and bio cluster apart would mean weight space
is organized by *capability family* rather than by task label — a more
interesting result than the original diagonal hypothesis. Every delta
helping everything would mean post-training reveals a generic
safe-to-edit region. A clean diagonal across distant domains would
finally support task-specific geometry.

### Screen with geometry before spending on behaviour

`subspace_overlap.py` costs minutes; a behavioural sweep costs hours.
Build the overlap matrix over candidate deltas first. A domain whose
subspace sits near the chance floor against math and code is worth a
behavioural sweep. One that lands on top of them will only reproduce the
result above, and the geometry already said so.

### Any delta source must be full-rank

**LoRA bakes in the answer.** A rank-r LoRA delta has rank exactly r. If
the adapter rank is at or below `BASIS_RANK`, the "leading subspace" of
that delta *is* the adapter's span — a hyperparameter someone picked,
not a discovered fact about where adaptation lands. Since the entire
premise here is learning *where* post-training writes, a delta whose
support was imposed by the training method is evidence about the method
and nothing else.

This rules out most community finetunes, which are overwhelmingly LoRA.
`Qwen2.5-7B-Medicine` was evaluated and rejected on exactly this ground,
despite having unusually good provenance otherwise: its card reports
fine-tuning directly from Qwen2.5-7B-Instruct, which would have made its
delta cleaner than Math's or Coder's. It is LoRA, so it is out.

Any source added to `subspace_overlap.py` has to come from full
fine-tuning, or from a LoRA of rank far above `BASIS_RANK` so that
extracting 32 is a genuine spectral compression rather than a copy.

### The checkpoint problem forces you to train your own

Public full-rank specialists do not extend past math and code at this
scale. On `mlx-community` at 4-bit 7B there is Math, Coder, and the plain
base. Elsewhere on the hub the Qwen2.5-7B translation and medical
finetunes are community LoRAs with unrelated data, recipes and durations
— a worse provenance confound than the one being escaped, and low-rank
on top of it.

So the controlled version has to be trained: one theta_0, one recipe,
only the dataset changing, and ideally several seeds per domain so that
"stable across independent runs of the same capability" can be separated
from "shared across all post-training".

**At 7B that means a high-rank adapter, not full fine-tuning.** Adam on a
7B needs roughly 84 GB for weights, gradients and optimizer state, so
full fine-tuning does not fit in 32 GB. A rank-256 LoRA does: the base
stays 4-bit frozen at ~4.2 GB, and the adapters plus optimizer state come
to roughly 12 GB. Extracting rank 32 from a rank-256 delta is then a real
compression, 32 directions chosen out of 256 by the spectrum rather than
handed over whole.

The honest limitation is that the delta is still confined to the
adapter's 256-dimensional span, so this measures where adaptation lands
*given* that constraint. That is a far weaker assumption than a rank-16
adapter, and it keeps the experiment at a size where the perturbation
results mean something.

**A cleaner alternative: full fine-tuning of a subset of layers.** The
analysis is per-matrix anyway, so full-rank deltas on four layers are
better evidence about where adaptation lands in *those* matrices than
low-rank deltas on all twenty-eight. Four of 28 layers is about 0.93B
parameters, roughly 11 GB with Adam, which fits. It trades coverage for
an unconstrained spectrum, and for a question that is specifically about
the shape of the delta's spectrum, that is the right trade. Running both
and checking they agree would be stronger still.

## Stage 2a: does the shared region have task-specific addresses?

`python stage2a_optima.py` -- roughly 80 minutes.

Stage 1 left one question open. The region is real and task-family
specific, but random points in it are worse than isotropic at finding
large gains. A region can be worth knowing even when random points in it
are not: "the restaurant is in San Francisco" narrows the search
enormously and still leaves you lost at a random street corner.

So: optimise an address directly for math, optimise one directly for
code, and ask whether they are different addresses. Both searches get
the rewards handed to them, so this is the *oracle* -- a ceiling on what
any navigator could predict. If directed search with the answers in hand
cannot find a good math address, no predictor can.

An address is 24 numbers. The raw parameterisation (dW = U M V^T, 32x32
across 196 matrices) is 200,704 coefficients, which no gradient-free
method searches in a few hundred evaluations, so `dictionary.py` draws
24 fixed directions inside the subspace once and an address is the
weight vector over them. That is also the shape a navigator would have
to emit, so the parameterisation is the real one.

### Every address is held at the same displacement

This is the control the experiment rests on, and it took four attempts
to get right. Measured facts that forced the design:

- At one fixed `||dW||/||W||`, different addresses produce KL from
  **0.016 to 0.186** -- an 11x spread. Comparing addresses at equal
  Frobenius norm compares perturbation strength, not direction.
- The log-log slope of KL against scale runs from **0.87 to 3.35**
  depending on address and scale, so no single global exponent can
  convert a measured KL into a scale correction.

The curves are monotonic, so each address fits its own slope by secant
from its own two measurements, iterating until it lands within 12% of
the target. Achieved spread across candidates is now 1.28x-1.45x,
comfortably under the 2.9x that changed which arm looked better in
stage 1.

Two earlier versions of this control failed silently and both produced
plausible-looking positives. Capturing the reference logits while a
perturbation was installed gave a cosine of -0.429 and a large
cross-gap, which reads exactly like "distinct optima, hypothesis
confirmed". Calibrating on math prompts alone left code displacement 5x
off target. The `kl`, `kl_calibration_probe` and `histories` fields in
the output exist to make that class of failure visible.

### Reading the output

Held-out margin change for four addresses -- math-optimised,
code-optimised, and the best of 30 random ones per family:

- **Both cross-gaps positive, cosine low** -- different tasks want
  different coordinates in one shared map. Stage 2b is on.
- **Gaps near zero, cosine high** -- both searches walked to the same
  point. Nothing task-specific to predict.
- **Neither beats best-random** -- 120 directed evaluations found
  nothing 30 random draws did not. The region is flat.
- **Held-out much worse than search** -- the address overfitted 8 tasks;
  raise `SPLIT` before reading anything else.

Given stage 1 found random directions here produce *smaller* changes
than isotropic, "flat region" is a live outcome rather than a strawman.
That is the point: an hour to find out, against weeks of building a
navigator with nothing to predict.

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
