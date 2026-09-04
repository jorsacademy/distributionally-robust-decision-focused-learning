# Frozen Experiment Protocol

## Training data

The default training corpus mixes three in-distribution observations with one mild-stress observation. Validation uses disjoint seeds and a smaller nominal/stress mixture. The KL adversary can therefore emphasize rare high-loss observations without seeing any held-out test scenario.

## Models

All learned methods use the same MLP architecture, normalizers, optimizer class, and graph oracle.

1. `mse_erm`: empirical mean squared edge-cost error.
2. `mse_kl_dro`: KL-robust aggregation of per-observation MSE.
3. `spo_erm`: nominal mean SPO+ after an MSE warm start.
4. `spo_kl_dro`: KL-robust aggregation of per-observation SPO+ after the same warm start.

The robust radius is warmed from zero to its configured value. Validation and evaluation data never influence adversarial training weights.

## Evaluation scenarios

Disjoint deterministic seed ranges are used for:

- in-distribution;
- context mean shift;
- context covariance/correlation shift;
- heavy-tailed cost noise;
- a corridor-specific structural cost shift;
- a changed nonlinear context-cost mechanism;
- a combined shift.

Each scenario is reported separately.

## Primary metrics

- mean, median, P90, CVaR90, CVaR95, and maximum realized regret;
- regret under true conditional mean costs;
- exact path hit rates;
- prediction RMSE and MAE;
- KL-robust regret profiles over several radii;
- adversarial effective sample size and maximum observation weight;
- paired deterministic bootstrap confidence intervals;
- path feasibility and unique decision count.

No single weighted score combines these measurements.
