# Exactness and Reliability Contract

## Exact components

For the declared layered acyclic graph:

- shortest paths are solved by deterministic dynamic programming;
- every returned binary edge vector is independently checked for source-sink flow conservation;
- small graphs can enumerate every source-sink path and compare the global optimum with dynamic programming;
- realized regret is evaluated against the exact path under the realized cost vector;
- conditional regret is evaluated against the exact path under the generator's conditional mean vector.

For a finite empirical loss vector and forward KL ambiguity set:

- the maximizing law is an exponential tilt in the interior;
- its scalar dual temperature is found by bisection;
- weights, normalization, KL divergence, primal objective, and dual objective are checked;
- at a concentration boundary, the exact maximum-loss support is used;
- the autograd rule returns the maximizing probability vector as a Danskin subgradient.

## Approximate components

- the neural context-to-cost predictor is approximate;
- finite datasets approximate the underlying contextual distribution;
- bootstrap intervals have Monte Carlo error;
- synthetic distribution shifts are controlled experiments, not deployment guarantees;
- SPO+ is a convex surrogate for decision regret, not the regret itself.

## Ambiguity semantics

The KL ball is over the empirical distribution of training observations. It allows an adversary to reweight difficult context-cost observations. It is not:

- a Wasserstein ball around each individual edge-cost vector;
- a robust shortest-path problem with interval edge costs;
- a certified guarantee against arbitrary covariate shift;
- a learned ambiguity-set geometry.

This scope is stated explicitly to prevent different robust-learning formulations from being conflated.
