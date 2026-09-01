# Frozen V4 — Detailed Results

Numbers in this file are the detailed appendix to the summary tables in
`README.md`. Every value is taken from the frozen evidence bundle in
`results/frozen_v4/`, which ships with a SHA-256 manifest:

```bash
cd results/frozen_v4 && sha256sum -c SHA256SUMS.txt
```

Earlier frozen sets are superseded. Frozen V2's ranking metrics were
affected by optimistic tie handling; Frozen V3 fixed that but selected
fusion weights on an 800-user validation subsample, which made the
selection step overfit. Both defects are documented in the
*Evaluation Audit* section of `README.md`.

## Protocol

Dataset: MovieLens-1M (not redistributed; see `data/README.md`).

| | |
|---|---|
| ratings / users / movies | 1,000,209 / 6,040 / 3,883 |
| train / validation / test | 988,129 / 6,040 / 6,040 |
| split | chronological per user, leave-one-out |
| validation target | penultimate interaction |
| test target | final interaction |
| seeds | 42, 43, 44, 45, 46 (canonical: 42) |
| primary protocol | full-candidate Top-10 ranking |
| secondary protocol | 1 positive + 99 sampled negatives |
| tie policy | score descending, then item index ascending |
| fusion search | validation only, **all 6,040 users**, 0.1-step grid |
| popularity alpha | 0.10 |
| NeuMF max epochs | 25, early stop on validation HR@10 (patience 2) |
| device | CPU |
| SBERT | disabled |

## Canonical run (seed 42), full-candidate ranking

The canonical Frozen V4 artifact is the **Mac CPU seed-42 run**.

| Model | HR@10 | nDCG@10 | MRR | Coverage | Gini |
|---|---:|---:|---:|---:|---:|
| Random | 0.002318 | 0.001002 | 0.002120 | 1.000000 | 0.146799 |
| Content (TF-IDF) | 0.011755 | 0.005716 | 0.007101 | 0.573783 | 0.871696 |
| ItemKNN (rating) | 0.005629 | 0.002730 | 0.004552 | 0.423384 | 0.923819 |
| Biased MF | 0.022185 | 0.010271 | 0.010802 | 0.277878 | 0.958034 |
| Popularity | 0.036921 | 0.018033 | 0.019903 | 0.051249 | 0.993279 |
| Item-CF (implicit) | 0.076159 | 0.037820 | 0.037240 | 0.323461 | 0.916411 |
| NeuMF | 0.079139 | 0.039406 | 0.039287 | 0.452485 | 0.879955 |
| **Hybrid** | **0.081126** | **0.039155** | **0.038264** | 0.439866 | 0.878857 |

Under leave-one-out evaluation, every user contributes exactly one
held-out relevant item. HR@10 is the direct Top-10 hit measure, while
nDCG@10 and MRR additionally reward better positions in the ranked list.

## Five-seed summary

| System | HR@10 | nDCG@10 | MRR | Coverage |
|---|---:|---:|---:|---:|
| Random | 0.002517 ± 0.000932 | 0.001063 ± 0.000427 | 0.002195 ± 0.000337 | 1.000000 |
| Content (TF-IDF) | 0.011755 | 0.005716 | 0.007101 | 0.573783 |
| ItemKNN (rating) | 0.005629 | 0.002730 | 0.004552 | 0.423384 |
| Biased MF | 0.020331 ± 0.002112 | 0.009650 ± 0.000928 | 0.010630 ± 0.000619 | 0.308885 ± 0.027740 |
| Popularity | 0.036921 | 0.018033 | 0.019903 | 0.051249 |
| Item-CF (implicit) | 0.076159 | 0.037820 | 0.037240 | 0.323461 |
| NeuMF | 0.076623 ± 0.002125 | 0.038415 ± 0.000801 | 0.038806 ± 0.001123 | 0.444399 ± 0.028502 |
| **Hybrid** | **0.083113 ± 0.001350** | **0.040153 ± 0.000696** | **0.039153 ± 0.000723** | 0.413392 ± 0.017109 |

### Paired comparison, Hybrid vs strongest single model of each seed

| Seed | Best single | Best single HR@10 | Hybrid HR@10 | Δ HR@10 | Δ nDCG@10 |
|---:|---|---:|---:|---:|---:|
| 42 | NeuMF | 0.0791 | 0.0811 | +0.0020 | -0.0003 |
| 43 | NeuMF | 0.0772 | 0.0826 | +0.0055 | +0.0030 |
| 44 | Item-CF (implicit) | 0.0762 | 0.0834 | +0.0073 | +0.0021 |
| 45 | Item-CF (implicit) | 0.0762 | 0.0848 | +0.0086 | +0.0032 |
| 46 | NeuMF | 0.0773 | 0.0836 | +0.0063 | +0.0012 |

Mean Δ HR@10 = **+0.005927 ± 0.002496**,
with Hybrid ahead in **5/5 seeds**.

Mean Δ nDCG@10 = **+0.001859 ± 0.001415**,
with Hybrid ahead in **4/5 seeds**.

### Comparison with Frozen V3

Frozen V3 and Frozen V4 should not be treated as a one-variable
controlled comparison.

Frozen V4 both:

1. replaces the 800-user fusion-search subsample with all 6,040
   validation users; and
2. includes the corrected bounded NeuMF negative-sampling implementation.

A separate fixed-score diagnostic isolates validation-user subsampling
and shows that this factor alone is sufficient to generate substantial
fusion-selection variance.

| Metric | Frozen V3 | Mac Frozen V4 |
|---|---:|---:|
| Hybrid HR@10 | 0.077947 ± 0.005715 | **0.083113 ± 0.001350** |
| Hybrid nDCG@10 | 0.037533 ± 0.003445 | **0.040153 ± 0.000696** |
| Hybrid HR wins vs best single | no mean advantage | **5/5 seeds** |
| Mean paired Δ HR@10 | -0.000497 | **+0.005927** |

Because the V4 methodology revision was made after earlier test results
had already been inspected, this is an engineering benchmark revision,
not a fresh unbiased holdout comparison.

## Sampled protocol (1 positive + 99 negatives), five-seed means

| System | HR@10 | nDCG@10 | AUC |
|---|---:|---:|---:|
| Random | 0.098344 | 0.044308 | 0.501331 |
| Content (TF-IDF) | 0.198113 | 0.103066 | 0.528413 |
| ItemKNN (rating) | 0.223974 | 0.100891 | 0.628767 |
| Biased MF | 0.273245 | 0.145931 | 0.625161 |
| Popularity | 0.472152 | 0.261052 | 0.822050 |
| Item-CF (implicit) | 0.667086 | 0.391877 | 0.891191 |
| NeuMF | 0.694139 | 0.414196 | **0.904780** |
| **Hybrid** | **0.698675** | **0.420693** | 0.879928 |

The sampled protocol is secondary. It is useful for diagnostic
comparability, but it is not interchangeable with the primary
full-candidate ranking protocol.

## Fusion weights

| Seed | ItemKNN | Item-CF | MF | NeuMF | Content |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.0 | 0.1 | 0.1 | 0.6 | 0.2 |
| 43 | 0.0 | 0.2 | 0.0 | 0.7 | 0.1 |
| 44 | 0.0 | 0.2 | 0.0 | 0.6 | 0.2 |
| 45 | 0.0 | 0.2 | 0.0 | 0.6 | 0.2 |
| 46 | 0.0 | 0.2 | 0.0 | 0.6 | 0.2 |

ItemKNN receives zero weight in all five Mac seeds.

Biased MF receives zero weight in four seeds and 0.1 only in seed 42.

The dominant contributors are NeuMF, Item-CF and Content.

The result supports a benchmark-specific distinction between explicit
rating-prediction quality and incremental Top-N ranking utility. It is
not a universal claim about rating-based models.

## Rating prediction (test split, no leakage)

| Model | MAE | MSE | RMSE | NMAE | NRMSE |
|---|---:|---:|---:|---:|---:|
| Global mean (baseline) | 0.9893 | 1.3820 | 1.1756 | 0.2473 | 0.2939 |
| ItemKNN | 0.7382 | 0.9157 | 0.9569 | 0.1846 | 0.2392 |
| **Biased MF** | **0.7175** | **0.8250** | **0.9083** | **0.1794** | **0.2271** |

Biased MF is the best rating predictor here and receives zero fusion
weight for ranking. NeuMF is trained on implicit feedback and outputs
interaction probabilities, so it is excluded from rating metrics by
design rather than omitted.

## Popularity / exposure trade-off (seed 42)

Mac canonical seed-42 artifact:

| Setting | HR@10 | nDCG@10 | Coverage | Diversity | Novelty | Gini |
|---|---:|---:|---:|---:|---:|---:|
| alpha=0.0 | 0.0808 | 0.0394 | 0.4066 | 0.6894 | 2.4465 | 0.8923 |
| alpha=0.05 | 0.0808 | 0.0391 | 0.4218 | 0.6852 | 2.4940 | 0.8860 |
| alpha=0.1 | 0.0811 | 0.0392 | 0.4399 | 0.6804 | 2.5469 | 0.8789 |
| alpha=0.2 | 0.0805 | 0.0384 | 0.4759 | 0.6718 | 2.6557 | 0.8637 |
| alpha=0.4 | 0.0760 | 0.0362 | 0.5555 | 0.6524 | 2.9159 | 0.8254 |
| 原实现公式 score*(1-0.2*pop) | 0.0023 | 0.0011 | 0.3178 | 0.6114 | 2.9642 | 0.9098 |

Alpha = 0.10 remains the pre-existing default. It was not retuned after
Frozen V4 test inspection.

## Controlled reproduction of the legacy defects (seed 42)

Same data split, same trained models; only the two legacy defects are
toggled.

| Popularity defect | Candidate defect | Version | P@10 | HR@10 | nDCG@10 | AUC | Positive in candidates |
|:---:|:---:|---|---:|---:|---:|---:|---:|
| off | off | current | 0.0067 | 0.0671 | 0.0310 | 0.8285 | 100% |
| on | off | control | 0.0000 | 0.0002 | 0.0002 | 0.1545 | 100% |
| off | on | control | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0%** |
| on | on | legacy | **0.0000** | **0.0000** | **0.0000** | **0.8677** | **0%** |

The bottom row reproduces the legacy behaviour: five ranking metrics
pinned at exactly zero while AUC reads 0.8677. AUC is high only when
*both* defects are active — with the candidate defect alone it is 0.0000.
The mechanism is that an excluded positive falls back to a default score
of 0 while negatives, scaled by the inverted popularity term, are
negative. This is a property of the defects, not of any model, and does
not change with more compute.

## Runtime

Runtime is environment-dependent and is therefore not used as a
cross-platform headline benchmark.

The canonical Frozen V4 artifact was produced on macOS / Apple Silicon
using CPU execution.

Full-validation fusion search is intentionally more expensive than the
earlier 800-user search because it evaluates all 6,040 validation users
across the full weight grid.

Per-stage timing metadata for the canonical seed-42 run is preserved in
`results/frozen_v4/metrics_seed42.json`.

## Cross-platform reproducibility

A separate Linux CPU replication is preserved locally as an audit
reference, while the public Frozen V4 canonical artifact is the Mac CPU
run.

| System / metric | Linux reference | Mac canonical |
|---|---:|---:|
| Hybrid HR@10 | 0.083940 | 0.083113 |
| Hybrid nDCG@10 | 0.040986 | 0.040153 |
| NeuMF HR@10 | 0.077815 | 0.076623 |
| NeuMF nDCG@10 | 0.039022 | 0.038415 |

The deterministic classical components reproduced consistently.

NeuMF does not reproduce bit-for-bit because small floating-point
differences alter validation trajectories and therefore early-stopping
checkpoints. Hybrid also moves slightly because it includes NeuMF.

The qualitative Frozen V4 conclusion remained consistent across the two
environments.

The repository therefore distinguishes **reproducible procedure** from
**bit-for-bit neural reproducibility** rather than claiming the latter.
