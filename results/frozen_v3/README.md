# Frozen V3

Frozen V3 is the canonical audited benchmark for this repository.

## What changed from Frozen V2

Frozen V2 used an optimistic exact-score tie policy: the held-out
positive was assigned the best possible position inside a tie block.

A targeted ItemKNN diagnostic found:

- 115 / 115 Top-10 hits were assigned rank 1
- 115 / 115 were inside exact-score tie blocks
- median tie-block size = 72
- maximum tie-block size = 666

Frozen V3 uses deterministic ranking everywhere:

1. score descending
2. internal item index ascending for exact ties

The correction is covered by regression tests.

## Canonical protocol

- MovieLens-1M
- 6,040 users
- 3,883 movies
- 1,000,209 ratings
- chronological train / validation / test holdout
- one validation and one test interaction per user
- full-candidate Top-10 ranking as the primary protocol
- sampled 1-positive + 99-negative metrics as secondary diagnostics
- canonical seed = 42
- NeuMF maximum epochs = 25
- NeuMF training negatives = 4 per positive
- validation-only fusion weight search
- popularity alpha = 0.10
- SBERT disabled

## Five-seed pipeline robustness

Seeds: 42, 43, 44, 45, 46.

| System | HR@10 | nDCG@10 | MRR | Coverage |
|---|---:|---:|---:|---:|
| Item-CF | 0.076159 | 0.037820 | 0.037240 | 0.323461 |
| NeuMF | 0.078444 ± 0.001155 | 0.039352 ± 0.000641 | 0.039417 ± 0.000403 | 0.456606 ± 0.011687 |
| Hybrid | 0.077947 ± 0.005715 | 0.037533 ± 0.003445 | 0.036851 ± 0.003854 | 0.401545 ± 0.031280 |

NeuMF had the strongest mean full-candidate ranking performance among
the evaluated individual models and remained much more stable than the
validation-selected Hybrid.

The Hybrid occasionally improved HR@10 for individual seeds, but it
did not provide a robust ranking advantage. Its fusion weights also
varied substantially across seeds.

This is an end-to-end pipeline robustness check, not a pure neural
initialization study: the seed affects stochastic model training and
the sampled validation users used for fusion-weight search.

## Interpretation boundary

Biased MF is optimized for explicit-rating regression, while NeuMF is
optimized for implicit Top-N ranking. Differences between them should
be interpreted as objective mismatch, not as a controlled proof that
one architecture is intrinsically superior.

Likewise, the five-seed results are descriptive robustness evidence.
They are not presented as a formal statistical significance test.

## Evidence files

- `metrics_seed42.json` — canonical seed-42 metrics and configuration
- `frozen_v3_multiseed_raw.csv` — per-seed metrics
- `frozen_v3_multiseed_summary.csv` — mean / standard deviation summary
- `frozen_v3_fusion_weights.csv` — validation-selected weights by seed
- `ablation_popularity.csv` — seed-42 popularity trade-off
- `ablation_bugs.csv` — controlled legacy-bug comparison
- `tie_audit.txt` — Frozen V2 tie-ranking diagnosis
- `pytest.txt` — regression / invariant test result
- `environment.txt` — execution environment
- `SHA256SUMS.txt` — artifact checksums

Frozen V2 remains in repository history for auditability, but its
ranking metrics are superseded by Frozen V3.
