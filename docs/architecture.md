# Architecture

## Data path

```text
context x
   │
   ▼
shared MLP cost predictor
   │
   ▼
predicted edge costs c_hat
   │
   ├── exact DAG shortest-path oracle
   │       └── deployed path decision
   │
   └── SPO+ oracle call during training
           ├── shortest path under true c
           └── shortest path under 2 c_hat - c
```

The model never emits a path directly. It predicts edge costs; every decision is produced by the same exact dynamic-programming optimizer.

## Robust aggregation

For per-observation losses `l_i`, the KL adversary solves

\[
\sup_{p\in\Delta_N}
\sum_i p_i l_i
\quad\text{s.t.}\quad
D_{KL}(p\|q)\leq\rho.
\]

The forward pass solves the one-dimensional dual to numerical tolerance. The backward pass returns the exact maximizing weights, which is a valid Danskin subgradient of the robust empirical risk.

## Separation of responsibilities

- `domain.py`: graph, exact decisions, path audits.
- `generator.py`: contextual cost mechanism and shifts.
- `dataset.py`: versioned exact-labeled corpora.
- `losses.py`: prediction loss, SPO+, regret, CVaR.
- `ambiguity.py`: KL ambiguity solver and robust autograd aggregation.
- `model.py`: shared predictor and safe checkpoints.
- `training.py`: four controlled training objectives.
- `evaluation.py`: prediction, regret, tail, ambiguity, and bootstrap metrics.
- `experiment.py`: frozen multi-shift protocol.
