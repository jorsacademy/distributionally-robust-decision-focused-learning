# Model Card

## Intended use

Methodological experiments on prediction-focused, decision-focused, and distributionally robust learning for a fixed layered-DAG shortest-path problem.

## Inputs and outputs

Input: a finite real-valued context vector. Output: one predicted cost per graph edge. An exact optimizer converts costs to a path.

## Training objectives

The repository supports MSE and SPO+, each with either empirical-mean or KL-DRO aggregation. Robustness is applied across empirical observations.

## Known limitations

- fixed graph topology and edge dimension;
- synthetic contextual costs;
- no guarantee under arbitrary distribution shift;
- no within-observation Wasserstein or moment ambiguity;
- no learned ambiguity radius;
- no general graph neural network;
- no claim of calibrated edge-cost uncertainty;
- no claim that KL-DRO always improves average or tail regret.

Negative transfer, higher conservatism, unstable path choices, and worse nominal performance remain reportable outcomes.
