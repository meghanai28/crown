CROWN

Can we learn a low-dimensional coordinate system of a pretrained
model's local task landscape, then use a task description or a few
examples to navigate directly toward a specialist, including unseen
task compositions?

This repository currently implements the random-direction
baseline. Random perturbations are not the proposed final navigator.
They establish whether nearby, behaviorally distinct models can be
measured before a learned direction predictor is added.

Current experiment

model.py loads one frozen MLX-LM checkpoint and performs
deterministic generation.

tasks.py defines development and held-out test tasks.

scoring.py separates answer-format compliance, exact typed value,
and semantic computation.

perturb.py creates one reproducible, normalized, multi-layer
low-rank direction.

verify_perturbation.py checks logits before running an expensive
search.

select_specialists.py selects only on development tasks and
evaluates the winner once on held-out test tasks.

For every targeted weight matrix, the perturbation scale means

[
\frac{\lVert \Delta W \rVert_F}{\lVert W \rVert_F}
= |\text{scale}|.
]

The same seed at different scales is the same direction at different
distances. Different seeds are different directions.

Run order

python scoring.py
python verify_perturbation.py
python select_specialists.py

Do not interpret unchanged greedy text as proof of an inactive
perturbation. First inspect the logit changes printed by
verify_perturbation.py.

Project layout

crown/
├── model.py
├── tasks.py
├── scoring.py
├── perturb.py
├── verify_perturbation.py
├── select_specialists.py
├── results/
│   └── selected_specialists.json   # created by the search
└── README.md

The next research stage should replace random selection with a dense
objective (such as expected-answer log probability), collect useful
directions, and learn a navigator that predicts or composes direction
coefficients from task evidence.