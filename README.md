# Distributionally Robust Decision-Focused Learning

[![CI](https://github.com/jorsacademy/distributionally-robust-decision-focused-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/distributionally-robust-decision-focused-learning/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research implementation of **distributionally robust decision-focused learning (DR-DFL)** for contextual shortest-path decisions.

The repository asks a narrower and more auditable question than “does robust AI perform better?”:

> When edge-cost predictors are trained with a KL-distributionally robust aggregation of the SPO+ decision surrogate, how do average regret, tail regret, adversarial sample weights, and generalization change under controlled distribution shifts?

Every downstream decision is produced by an exact dynamic-programming shortest-path oracle. The robust empirical-risk layer is solved by an exact one-dimensional KL dual. Prediction error, decision regret, tail behavior, ambiguity sensitivity, and feasibility are reported separately.

## Claims boundary

This is a compact methodology benchmark. It does **not** claim:

- universal robustness to out-of-distribution data;
- state-of-the-art shortest-path prediction or optimization;
- reproduction of differentiable mixed-integer DRO layers;
- a Wasserstein ambiguity set around each edge-cost vector;
- a learned ambiguity-set geometry or radius;
- robustness guarantees for arbitrary covariate shifts;
- superiority of KL-DRO on every nominal or shifted distribution;
- industrial-scale graph support.

The implemented ambiguity set reweights the empirical distribution of complete context-cost observations. That scope is deliberate and documented.

## Contextual optimization problem

A fixed layered directed acyclic graph has edge set \(E\). For context \(x\), the unknown true edge-cost vector is \(c(x)\). A predictor \(f_\theta\) produces

\[
\hat c=f_\theta(x).
\]

The downstream optimizer selects a source-to-sink path:

\[
w^*(\hat c)
\in
\arg\min_{w\in\mathcal S}
\hat c^\top w,
\]

where \(\mathcal S\subset\{0,1\}^{|E|}\) is the source-to-sink path set. The realized decision regret is

\[
L_{\mathrm{SPO}}(\hat c,c)
=
c^\top w^*(\hat c)-c^\top w^*(c).
\]

Low edge-cost MSE does not necessarily imply low path regret: errors on edges that cannot change the chosen path may be harmless, while smaller errors near a path-switching boundary can be operationally expensive.

## SPO+ structured surrogate

Directly optimizing path regret is difficult because the optimizer map is piecewise constant. The repository implements the SPO+ surrogate:

\[
L_{\mathrm{SPO+}}(\hat c,c)
=
\max_{w\in\mathcal S}
(c-2\hat c)^\top w
+2\hat c^\top w^*(c)
-c^\top w^*(c).
\]

Using a minimization oracle, define

\[
\tilde w=w^*(2\hat c-c).
\]

Then a valid subgradient with respect to the prediction is

\[
\partial_{\hat c}L_{\mathrm{SPO+}}
=2\left(w^*(c)-\tilde w\right).
\]

The custom autograd operator therefore uses two exact path decisions:

```text
true edge costs c ───────────────► exact path w*(c)

2 c_hat - c ─────────────────────► exact path w*(2 c_hat - c)

subgradient = 2 [w*(c) - w*(2 c_hat - c)]
```

The training code does not differentiate through a relaxed path polytope or replace the graph problem with a softmax proxy.

## KL-distributionally robust empirical risk

For per-observation losses \(\ell_1,\ldots,\ell_N\), nominal empirical probabilities \(q_i=1/N\), and radius \(\rho\), the robust layer solves

\[
\mathcal R_\rho(\ell)
=
\sup_{p\in\Delta_N}
\sum_{i=1}^N p_i\ell_i
\]

subject to

\[
D_{\mathrm{KL}}(p\|q)
=
\sum_i p_i\log\frac{p_i}{q_i}
\leq\rho.
\]

For an interior solution, the maximizing distribution is an exponential tilt:

\[
p_i(\eta)
=
\frac{q_i\exp(\ell_i/\eta)}
{\sum_j q_j\exp(\ell_j/\eta)}.
\]

The scalar temperature \(\eta>0\) is chosen so that

\[
D_{\mathrm{KL}}(p(\eta)\|q)=\rho.
\]

The code finds \(\eta\) by stable bisection and checks the primal value against the dual expression

\[
\eta\rho
+
\eta\log\left(
\sum_iq_i\exp(\ell_i/\eta)
\right).
\]

By Danskin's theorem, the maximizing weights \(p^*\) form a valid gradient of the robust risk with respect to the loss vector. The autograd layer returns those exact weights in its backward pass.

As \(\rho\) increases, the adversary concentrates more probability on high-loss observations. The benchmark reports effective sample size and maximum adversarial weight so that “robustness” is not treated as an opaque hyperparameter.

## DR-SPO training objective

The main method solves

\[
\min_\theta
\sup_{p\in\Delta_N:
D_{\mathrm{KL}}(p\|q)\leq\rho}
\sum_{i=1}^N
p_i
L_{\mathrm{SPO+}}
\left(f_\theta(x_i),c_i\right).
\]

A small prediction regularizer can be retained during decision-focused fine-tuning. The KL radius is linearly warmed from zero to its configured value to avoid beginning training with an immediately concentrated adversary.

## Controlled methods

All learned methods use the same MLP architecture, graph oracle, data, normalization, and optimizer family.

| Method | Per-observation loss | Aggregation | Purpose |
| --- | --- | --- | --- |
| `mse_erm` | edge-cost MSE | empirical mean | standard predict-then-optimize baseline |
| `mse_kl_dro` | edge-cost MSE | KL-DRO | separates robust prediction from robust DFL |
| `spo_erm` | SPO+ | empirical mean | nominal decision-focused learning |
| `spo_kl_dro` | SPO+ | KL-DRO | distributionally robust decision-focused learning |
| `ridge_prediction` | squared error fit | closed-form ridge | non-neural contextual baseline |
| `global_mean` | none | global training mean | context-free baseline |
| `true_mean_oracle` | generator conditional mean | exact path | conditional-information reference |
| `perfect_information_oracle` | realized costs | exact path | zero-realized-regret lower bound |

The SPO methods receive the same MSE warm-start budget. Robust and nominal variants are not given different graph oracles.

## Exact shortest-path core

The graph contains a source, a fixed number of hidden layers, and a sink. Consecutive layers are fully connected. Dynamic programming processes nodes in topological order and uses deterministic edge-index tie breaking.

For small graphs the repository independently enumerates all

\[
\text{width}^{\text{layer count}}
\]

source-to-sink paths and verifies that exhaustive and dynamic-programming objectives agree.

Every path decision is independently checked for:

- binary bounds and integrality;
- one unit of source outflow;
- one unit of sink inflow;
- zero flow imbalance at every intermediate node;
- objective consistency under the evaluated cost vector.

## Synthetic contextual data and shifts

A fixed nonlinear data-generating mechanism maps contexts to positive edge costs. It contains linear and sinusoidal context effects, correlated noise, and a reproducible graph-corridor structure.

The frozen protocol evaluates disjoint seeds for:

1. `in_distribution` — nominal contexts and noise;
2. `mean_shift` — shifted context mean;
3. `covariance_shift` — changed scale and correlation;
4. `heavy_tail` — Student-t cost noise;
5. `corridor_shift` — structural cost increase on one graph corridor;
6. `nonlinear_shift` — changed nonlinear context-cost response;
7. `combined_shift` — simultaneous context, structural, and noise shift.

Training contains mostly nominal observations plus a minority of mild-stress cases. Held-out shift regimes are never used to compute training adversarial weights.

## Evaluation

### Prediction quality

- edge-cost RMSE;
- edge-cost MAE.

### Realized decision quality

- mean and median regret;
- P90 regret;
- CVaR90 and CVaR95 regret;
- maximum regret;
- mean relative regret;
- exact realized-path hit rate.

### Conditional decision quality

The synthetic generator exposes its conditional mean only for audit. The benchmark measures regret and path hit rate against the exact path under that mean. Models are not trained on these mean labels.

### Distributional robustness diagnostics

- KL-robust regret at several evaluation radii;
- adversarial effective sample size;
- maximum adversarial observation weight;
- full ambiguity profile rather than one chosen radius;
- worst regime mean regret when a corpus contains multiple regimes.

### Statistical reporting

A deterministic paired bootstrap supplies a 95% interval for mean regret. The seed and number of bootstrap draws are stored in every report.

### Reliability

- exact path feasibility rate;
- unique path count;
- graph and corpus fingerprints;
- candidate and optimal realized cost.

No weighted composite score hides trade-offs among nominal regret, tail regret, prediction error, and adversarial concentration.

## Installation

```bash
python -m pip install -e ".[dev]"
```

CPU-only PyTorch is sufficient.

## CLI

### Generate one contextual instance

```bash
drdfl generate \
  --layers 4 \
  --width 4 \
  --context-dim 8 \
  --regime in_distribution \
  --seed 42 \
  --output artifacts/example.json
```

### Build deterministic corpora

```bash
drdfl collect \
  --count 128 \
  --layers 4 \
  --width 4 \
  --context-dim 8 \
  --regimes in_distribution in_distribution in_distribution mild_stress \
  --seed 1000 \
  --output artifacts/train.jsonl

drdfl collect \
  --count 40 \
  --layers 4 \
  --width 4 \
  --context-dim 8 \
  --regimes in_distribution mild_stress \
  --seed 2000 \
  --output artifacts/validation.jsonl
```

### Cross-check the exact path oracle

```bash
drdfl oracle artifacts/validation.jsonl \
  --sample-index 0 \
  --output artifacts/oracle-check.json
```

### Train nominal SPO+

```bash
drdfl train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --mode spo_erm \
  --epochs 120 \
  --warm-start-epochs 20 \
  --checkpoint artifacts/spo-erm.safetensors \
  --output-report artifacts/spo-erm-training.json
```

### Train KL-robust SPO+

```bash
drdfl train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --mode spo_kl_dro \
  --kl-radius 0.15 \
  --epochs 120 \
  --warm-start-epochs 20 \
  --checkpoint artifacts/spo-kl-dro.safetensors \
  --output-report artifacts/spo-kl-dro-training.json
```

### Compare checkpoints

```bash
drdfl benchmark artifacts/test.jsonl \
  --train-dataset artifacts/train.jsonl \
  --checkpoint spo_erm=artifacts/spo-erm.safetensors \
  --checkpoint spo_kl_dro=artifacts/spo-kl-dro.safetensors \
  --selected-radius 0.15 \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

### Run the frozen research protocol

```bash
drdfl research \
  --config configs/research_v1.json \
  --checkpoint-directory artifacts/checkpoints \
  --output-report artifacts/research-report.json
```

## Reproducibility and checkpoint safety

- graph structure, contexts, costs, shifts, training, bootstrap, and evaluation use explicit seeds;
- train, validation, and each shift scenario use disjoint seed ranges;
- JSONL manifests contain record counts, graph definitions, metadata, and stable SHA-256 fingerprints;
- corpus loading recomputes every stored exact path objective;
- checkpoints use Safetensors, not pickle;
- checkpoint metadata records model dimensions, training mode, graph fingerprint, corpus fingerprints, and training configuration;
- non-finite predictions, losses, gradients, or adversarial probabilities fail closed.

## Repository layout

```text
src/drdfl/
├── domain.py       # layered graph, exact path DP, flow audits
├── generator.py    # contextual costs and controlled shifts
├── dataset.py      # exact-labeled JSONL corpora and fingerprints
├── ambiguity.py    # exact KL adversary and Danskin autograd layer
├── losses.py       # MSE, SPO+, exact regret, CVaR
├── model.py        # shared MLP and Safetensors checkpoints
├── training.py     # ERM and KL-DRO training modes
├── baselines.py    # ridge and context-free controls
├── evaluation.py   # regret, tail, ambiguity, bootstrap metrics
├── oracle.py       # exhaustive path verification
├── experiment.py   # frozen seven-scenario protocol
└── cli.py          # end-to-end command-line workflows
```

Additional detail:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/exactness.md`](docs/exactness.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/research_context.md`](docs/research_context.md)
- [`docs/model_card.md`](docs/model_card.md)

## Tests and CI

GitHub Actions runs on Python 3.11 and 3.12:

```text
package installation and dependency check
Ruff lint
Ruff formatting check
strict mypy
branch-aware pytest coverage
collect → train DR-SPO → exact oracle → benchmark smoke
```

The regression suite covers exact path enumeration, flow audits, deterministic shifts, corpus tamper detection, SPO+ upper bounds and gradients, KL primal-dual consistency, Danskin gradients, all four training modes, safe checkpoint round trips, ambiguity profiles, research scenarios, and CLI workflows.

## Methodological limitations

The graph topology is fixed and the model predicts one cost for every edge. This is not a variable-topology graph neural network. The empirical KL ball can protect against reweightings of observed training cases, but it cannot create unseen contexts or guarantee performance under structural shifts outside the training support.

SPO+ is optimized because direct regret has a discontinuous decision map. Better SPO+ does not mechanically imply lower regret on every finite sample. Similarly, a larger KL radius can improve tail behavior while degrading nominal mean regret or causing excessive concentration on a few training observations. These are experimental outcomes, not bugs to hide.

The exact path oracle is inexpensive for a layered DAG. The repository studies learning methodology, not solver acceleration. No claim is made that neural prediction is faster than simply solving a path problem once true costs are known.

## Research context

The implementation is positioned relative to:

- Adam N. Elmachtoub and Paul Grigas, [“Smart Predict, then Optimize”](https://doi.org/10.1287/mnsc.2020.3922), *Management Science* 68(1), which introduced the SPO framework and SPO+ surrogate;
- Xutao Ma, Chao Ning, and Wenli Du, [“Differentiable Distributionally Robust Optimization Layers”](https://proceedings.mlr.press/v235/ma24j.html), ICML 2024, which develops differentiable DRO layers for contextual decision pipelines;
- Abhilash Reddy Chenreddy and Erick Delage, [“End-to-end Conditional Robust Optimization”](https://proceedings.mlr.press/v244/chenreddy24a.html), UAI 2024, which jointly trains contextual uncertainty and downstream decisions;
- Shoki Yamao et al., [“Robust Decision-Focused Learning via Worst-Case Regret Minimization”](https://proceedings.mlr.press/v337/yamao26a.html), UAI 2026, which studies measurement error and deployment distribution shift through robust regret objectives.

This repository does not reproduce those architectures. It supplies a smaller exact-oracle laboratory for empirical KL reweighting and SPO+.

## License

This repository is **source-available for non-commercial use** under the [PolyForm Noncommercial License 1.0.0](LICENSE). It is not described as OSI Open Source because commercial use is not granted.
