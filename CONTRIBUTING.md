# Contributing

Changes should preserve the distinction between prediction quality, decision quality, and distributional robustness.

## Required checks

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Methodological requirements

A change to the shortest-path oracle must retain exhaustive tiny-graph verification. A change to the ambiguity solver must include normalization, radius, primal-dual, and gradient tests. A new learning objective must be compared with prediction-focused and nominal decision-focused controls under identical data and model budgets.

Do not report only training loss. New experiments should preserve at least:

- realized regret and conditional-mean regret;
- tail regret and a KL ambiguity profile;
- path feasibility and exact-oracle comparisons;
- prediction RMSE/MAE;
- distribution-shift results by scenario;
- seeds, corpus fingerprints, and checkpoint metadata.

Avoid claims of guarantees that are not supplied by the exact path and KL-adversary components. In particular, empirical robustness under synthetic shifts is not a universal out-of-distribution guarantee.
