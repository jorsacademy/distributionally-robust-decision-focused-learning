# Security and Reliability

## Supported scope

This repository processes local JSON/JSONL corpora and Safetensors checkpoints. It does not execute generated code, shell fragments, serialized Python objects, or user-supplied plugins.

## Data and checkpoint handling

- Corpus loaders validate schema version, record count, exact path labels, and a SHA-256 fingerprint.
- Checkpoints use Safetensors rather than pickle-based deserialization.
- Checkpoint schema, feature schema, model dimensions, and graph metadata are validated before evaluation.
- Non-finite contexts, costs, predictions, losses, gradients, or ambiguity weights cause an error.

## Mathematical fail-closed behavior

The benchmark aborts when:

- a shortest-path decision fails flow or integrality checks;
- dynamic programming disagrees with exhaustive enumeration on a verification instance;
- an SPO+ value is materially negative;
- a candidate path appears better than the exact path oracle beyond tolerance;
- the KL adversary violates normalization, radius activity, or primal-dual consistency;
- a dataset fingerprint or stored exact objective is inconsistent.

## Operational limitations

The code is a research benchmark, not a safety-certified routing service. Synthetic distribution shifts do not establish robustness for deployment data. Before production use, independently review the graph model, data provenance, ambiguity-set semantics, numerical tolerances, privacy controls, and failure handling.

Report vulnerabilities privately through GitHub's security reporting mechanism rather than a public issue.
