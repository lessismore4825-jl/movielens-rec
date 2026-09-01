# Frozen V3 Results

Frozen V3 is the canonical audited benchmark for this repository.

Frozen V2 remains part of the repository history for auditability, but
its ranking metrics are superseded because its evaluator handled
exact-score ties optimistically.

## Protocol

Dataset: MovieLens-1M.

- ratings: 1,000,209
- users: 6,040
- movies: 3,883
- train interactions: 988,129
- validation interactions: 6,040
- test interactions: 6,040
- split: chronological per user
- validation target: penultimate interaction
- test target: final interaction
- canonical seed: 42
- NeuMF maximum epochs: 25
- NeuMF training negatives: 4 per positive
- Top-N: 10
- sampled protocol: 1 positive + 99 negatives
- fusion search: validation only, 800 users
- popularity alpha: 0.10
- SBERT: disabled

The primary evaluation protocol is full-candidate ranking over unseen
items. Sampled ranking is retained as a secondary diagnostic.

Exact-score ties use the same deterministic rule throughout the
pipeline:

1. score descending
2. internal item index ascending

## Canonical Full-Candidate Ranking

| Model | HR@10 | nDCG@10 | MRR | MedianRank | Coverage | Novelty | Gini |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ItemKNN | 0.0056 | 0.0027 | 0.0046 | 1086.0 | 0.4234 | 6.7152 | 0.9238 |
| Item-CF | 0.0762 | 0.0378 | 0.0372 | 181.5 | 0.3235 | 2.3052 | 0.9164 |
| Biased MF | 0.0222 | 0.0103 | 0.0108 | 1059.0 | 0.2779 | 4.2472 | 0.9580 |
| NeuMF | 0.0795 | 0.0398 | 0.0396 | 159.0 | 0.4757 | 2.5440 | 0.8699 |
| Content (TF-IDF) | 0.0118 | 0.0057 | 0.0071 | 1634.5 | 0.5738 | 5.6481 | 0.8717 |
| Popularity | 0.0369 | 0.0180 | 0.0199 | 414.0 | 0.0512 | 1.2831 | 0.9933 |
| Random | 0.0023 | 0.0010 | 0.0021 | 1812.5 | 1.0000 | 6.4929 | 0.1468 |
| Hybrid | 0.0796 | 0.0382 | 0.0374 | 157.0 | 0.4028 | 2.4493 | 0.8948 |

The canonical seed-42 run is evidence for one frozen run, not a claim
that the best single-seed system is universally superior.

## Five-Seed Pipeline Robustness

Seeds: 42, 43, 44, 45, 46.

| System | HR@10 | nDCG@10 | MRR | Coverage |
| --- | --- | --- | --- | --- |
| Item-CF | 0.076159 ± 0.000000 | 0.037820 ± 0.000000 | 0.037240 ± 0.000000 | 0.323461 ± 0.000000 |
| NeuMF | 0.078444 ± 0.001155 | 0.039352 ± 0.000641 | 0.039417 ± 0.000403 | 0.456606 ± 0.011687 |
| Hybrid | 0.077947 ± 0.005715 | 0.037533 ± 0.003445 | 0.036851 ± 0.003854 | 0.401545 ± 0.031280 |

NeuMF achieved the strongest mean full-candidate ranking performance
among the evaluated individual models:

- HR@10 = 0.078444 ± 0.001155
- nDCG@10 = 0.039352 ± 0.000641
- MRR = 0.039417 ± 0.000403
- Coverage = 0.456606 ± 0.011687

Hybrid achieved:

- HR@10 = 0.077947 ± 0.005715
- nDCG@10 = 0.037533 ± 0.003445
- MRR = 0.036851 ± 0.003854
- Coverage = 0.401545 ± 0.031280

The Hybrid therefore showed no robust ranking advantage over NeuMF and
introduced substantially larger pipeline-level variance.

This is an end-to-end robustness study rather than a pure neural
initialization experiment because the seed affects both stochastic
training and the validation-user sample used for fusion search.

## Sampled Ranking

| Model | HR@10 | nDCG@10 | MRR | AUC |
| --- | --- | --- | --- | --- |
| ItemKNN | 0.2248 | 0.1009 | 0.0886 | 0.6292 |
| Item-CF | 0.6684 | 0.3913 | 0.3224 | 0.8909 |
| Biased MF | 0.2748 | 0.1484 | 0.1319 | 0.6255 |
| NeuMF | 0.6917 | 0.4147 | 0.3448 | 0.9046 |
| Content (TF-IDF) | 0.1972 | 0.1021 | 0.0941 | 0.5280 |
| Popularity | 0.4730 | 0.2600 | 0.2175 | 0.8221 |
| Random | 0.1013 | 0.0438 | 0.0497 | 0.5061 |
| Hybrid | 0.6873 | 0.4136 | 0.3415 | 0.8609 |

Sampled evaluation produces much larger absolute ranking metrics than
full-candidate evaluation and can change the apparent relative strength
of different recommenders.

For that reason, Frozen V3 uses full-candidate ranking as the primary
benchmark.

Across five seeds, NeuMF sampled AUC was:

0.905311 ± 0.000531

The retired historical AUC 0.9127 is not comparable because it was
affected by the legacy candidate-set and popularity-scaling defects.

## Rating Prediction

Explicit-rating regression is evaluated separately from Top-N ranking.

| Model | MAE | RMSE |
|---|---:|---:|
| Global mean baseline | 0.9893 | 1.1756 |
| ItemKNN | 0.7382 | 0.9569 |
| Biased MF | 0.7175 | 0.9083 |

Biased MF is optimized for explicit rating prediction, whereas NeuMF is
optimized for implicit-feedback ranking with negative sampling and
Binary Cross-Entropy.

The gap between their Top-N metrics therefore demonstrates an
**objective mismatch**, not a controlled proof that one architecture is
intrinsically superior.

A cleaner architecture comparison would require an implicit-feedback
MF baseline such as BPR-MF or iALS.

## Fusion Weights

Validation-selected weights across seeds:

| Seed | ItemKNN | Item-CF | MF | NeuMF | Content |
| --- | --- | --- | --- | --- | --- |
| 42 | 0.2 | 0.1 | 0.1 | 0.5 | 0.1 |
| 43 | 0.2 | 0.2 | 0.2 | 0.1 | 0.3 |
| 44 | 0.1 | 0.2 | 0.2 | 0.4 | 0.1 |
| 45 | 0.0 | 0.1 | 0.1 | 0.6 | 0.2 |
| 46 | 0.0 | 0.2 | 0.0 | 0.7 | 0.1 |

The weights vary materially across seeds.

Paired Hybrid minus NeuMF differences:

| Metric | Mean Hybrid - NeuMF | SD |
| --- | --- | --- |
| HR@10 | -0.000497 | 0.005520 |
| nDCG@10 | -0.001819 | 0.003553 |
| MRR | -0.002565 | 0.004022 |
| Coverage | -0.055061 | 0.029327 |

These descriptive results do not support claiming that fusion
consistently improves ranking quality.

No formal statistical-significance claim is made from five seeds.

## Popularity / Exposure Ablation

Canonical seed-42 ablation:

| 设置 | Precision@10 | Recall@10 | F1@10 | nDCG@10 | HR@10 | MRR | MedianRank | Coverage | CoveredItems | Diversity | Novelty | Gini | Top1%ItemShare | EvalUsers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 修正公式 alpha=0.0 | 0.0079 | 0.0791 | 0.0144 | 0.0383 | 0.0791 | 0.0378 | 157.0000 | 0.3724 | 1446.0000 | 0.7094 | 2.3532 | 0.9067 | 0.3011 | 6040.0000 |
| 修正公式 alpha=0.05 | 0.0080 | 0.0796 | 0.0145 | 0.0385 | 0.0796 | 0.0378 | 158.0000 | 0.3873 | 1504.0000 | 0.7053 | 2.3996 | 0.9010 | 0.2880 | 6040.0000 |
| 修正公式 alpha=0.1 | 0.0080 | 0.0796 | 0.0145 | 0.0382 | 0.0796 | 0.0374 | 157.0000 | 0.4028 | 1564.0000 | 0.7009 | 2.4493 | 0.8948 | 0.2747 | 6040.0000 |
| 修正公式 alpha=0.2 | 0.0078 | 0.0781 | 0.0142 | 0.0372 | 0.0781 | 0.0365 | 158.0000 | 0.4394 | 1706.0000 | 0.6922 | 2.5538 | 0.8813 | 0.2494 | 6040.0000 |
| 修正公式 alpha=0.4 | 0.0073 | 0.0732 | 0.0133 | 0.0344 | 0.0732 | 0.0339 | 182.0000 | 0.5241 | 2035.0000 | 0.6730 | 2.8215 | 0.8446 | 0.1971 | 6040.0000 |
| 原实现公式 score*(1-0.2*pop) | 0.0002 | 0.0022 | 0.0004 | 0.0014 | 0.0022 | 0.0019 | 3534.0000 | 0.3049 | 1184.0000 | 0.6653 | 3.1349 | 0.9246 | 0.3640 | 6040.0000 |

The popularity experiment compares alpha values within the same Hybrid
system.

It therefore supports an accuracy / exposure trade-off interpretation,
but it should not be used to infer that Hybrid must achieve higher
Coverage than NeuMF.

Alpha = 0.10 remained the frozen default and was not retuned after
inspecting test results.

## Controlled Legacy-Bug Ablation

| 热度惩罚 bug | 候选集 bug | 对应版本 | Precision@10 | Recall@10 | F1@10 | nDCG@10 | HR@10 | AUC | Coverage | CoveredItems | Novelty | 正例在候选集内比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 关 | 关 | 本实现 | 0.0067 | 0.0666 | 0.0121 | 0.0312 | 0.0666 | 0.8257 | 0.3693 | 1434.0000 | 2.3504 | 1.0000 |
| 开 | 关 | 单 bug 对照 | 0.0000 | 0.0002 | 0.0000 | 0.0002 | 0.0002 | 0.1543 | 0.0577 | 224.0000 | 12.5602 | 1.0000 |
| 关 | 开 | 单 bug 对照 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3693 | 1434.0000 | 2.3527 | 0.0000 |
| 开 | 开 | 原实现 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8677 | 0.0577 | 224.0000 | 12.5602 | 0.0000 |

The controlled ablation demonstrates why the historical AUC could look
strong even when the held-out positive was structurally absent from the
candidate set.

Candidate construction and score scaling therefore had to be audited
before model quality could be interpreted.

## Exact-Score Tie Audit

Frozen V2 ItemKNN reported:

- HR@10 = 0.0190397351
- nDCG@10 = 0.0190397351

A targeted diagnostic found:

- total Top-10 hits = 115
- rank-1 hits = 115
- hits inside exact-score ties = 115 / 115
- unique positive scores = 0 / 115
- median tie-block size = 72
- maximum tie-block size = 666

The previous evaluator counted only candidates whose scores were
strictly greater than the positive score. That assigned the positive
the best possible position inside every exact-score tie block.

After applying one deterministic ranking rule everywhere, canonical
ItemKNN became approximately:

- HR@10 = 0.005629
- nDCG@10 = 0.002730

This is why Frozen V2 ranking metrics are superseded rather than silently
overwritten.

## Tests

The frozen codebase passes:

`19 passed`

Regression and invariant tests cover temporal split correctness,
candidate-set validity, information boundaries, ranking sanity checks,
popularity-penalty behavior, NeuMF negative sampling and deterministic
exact-score tie handling.

## Reproducibility

Canonical configuration:

`python run.py --data-dir data/ml-1m --out-dir outputs/reproduce_v3 --seed 42 --neumf-epochs 25 --ablation --bug-ablation`

Evidence is stored under:

`results/frozen_v3/`

The evidence bundle contains canonical metrics, multi-seed metrics,
fusion weights, controlled ablations, tie-audit evidence, environment
information, test output and SHA-256 checksums.

## Limitations

- Offline MovieLens evaluation is not evidence of production impact.
- Five seeds provide descriptive robustness evidence, not a formal
  significance study.
- Pipeline seed changes both stochastic training and the validation
  sample used during fusion search.
- Biased MF and NeuMF optimize different objectives.
- SBERT was not enabled in Frozen V3.
- No dedicated cold-start benchmark is included.
- No leave-one-signal-out experiment isolates the causal contribution
  of the Content component.

## Final Conclusions

1. Evaluation correctness was more important than adding model
   complexity.
2. NeuMF was the strongest mean individual full-candidate ranker across
   the five evaluated seeds.
3. Item-CF remained a strong deterministic classical baseline.
4. Validation-selected Hybrid did not demonstrate a robust advantage
   over NeuMF.
5. Sampled and full-candidate evaluation are not interchangeable.
6. Rating prediction and Top-N ranking should be treated as different
   objectives.
7. Candidate-set, popularity-scaling and exact-score tie defects were
   converted into explicit regression tests.
8. Frozen V3 is the final audited benchmark for this project; no further
   model tuning is performed after this freeze.
