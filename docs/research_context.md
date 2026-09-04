# Research Context

Decision-focused learning replaces a prediction-only objective with a loss informed by the downstream optimizer. The SPO/SPO+ framework provides a tractable structured surrogate for linear-objective optimization problems, including shortest paths.

Distributionally robust optimization replaces an empirical average with the worst expected loss over a neighborhood of the empirical distribution. This repository uses a forward-KL ball over observation weights because its adversary has an auditable exponential-tilting solution and an exact Danskin gradient.

The project is positioned relative to:

- Elmachtoub and Grigas, “Smart Predict, then Optimize,” *Management Science* 68(1), 2022, DOI: https://doi.org/10.1287/mnsc.2020.3922.
- Ma, Ning, and Du, “Differentiable Distributionally Robust Optimization Layers,” ICML 2024: https://proceedings.mlr.press/v235/ma24j.html.
- Chenreddy and Delage, “End-to-end Conditional Robust Optimization,” UAI 2024: https://proceedings.mlr.press/v244/chenreddy24a.html.
- Yamao et al., “Robust Decision-Focused Learning via Worst-Case Regret Minimization,” UAI 2026: https://proceedings.mlr.press/v337/yamao26a.html.

This implementation does not reproduce the differentiable mixed-integer DRO layer, conditional uncertainty-set learning, or Wasserstein worst-case-regret formulation in those papers. It isolates a smaller question: how does exact empirical-distribution reweighting interact with an exact SPO+ oracle and out-of-distribution path regret?
