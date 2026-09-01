# MovieLens-1M Recommender: Evaluation, Ranking & Trade-offs

A reproducible offline recommender-system study on MovieLens-1M covering
collaborative filtering, matrix factorization, neural ranking, content-based
recommendation, score fusion, popularity bias, and evaluation correctness.

## Dataset

- 1,000,209 ratings
- 6,040 users
- 3,883 movies
- Train: 988,129 interactions
- Validation: 6,040 interactions
- Test: 6,040 interactions

Raw MovieLens data is not committed. See `data/README.md`.

## Temporal Evaluation Protocol

Interactions are ordered chronologically for each user.

- Train: all interactions except the final two
- Validation: second-most-recent interaction
- Test: most-recent interaction

Training uses only training history.

Validation does not use future test information.

At test time, train and validation interactions are treated as observed
history while the test interaction remains the held-out target.

## Models

- Random baseline
- Popularity baseline
- ItemKNN
- implicit Item-CF
- Biased Matrix Factorization
- NeuMF
- TF-IDF content recommender
- validation-selected Hybrid fusion

NeuMF uses GMF + MLP, negative sampling, Binary Cross-Entropy, and
validation HR@10 early stopping.

SBERT is supported as an optional content encoder but was not used in
Frozen V2.

## Frozen V3 — Full-Candidate Ranking

**Frozen V3 is the canonical benchmark.**

Frozen V2 ranking metrics are retained only as audit history because
the previous evaluator handled exact-score ties optimistically.

Canonical protocol:

- MovieLens-1M
- chronological train / validation / test holdout
- canonical seed = 42
- NeuMF maximum epochs = 25
- full-candidate Top-10 ranking as the primary protocol
- deterministic tie-breaking:
  score descending, then internal item index ascending

### Five-seed pipeline robustness

Seeds: 42, 43, 44, 45, 46.

| System | HR@10 | nDCG@10 | MRR | Coverage |
|---|---:|---:|---:|---:|
| Item-CF | 0.076159 | 0.037820 | 0.037240 | 0.323461 |
| NeuMF | 0.078444 ± 0.001155 | 0.039352 ± 0.000641 | 0.039417 ± 0.000403 | 0.456606 ± 0.011687 |
| Hybrid | 0.077947 ± 0.005715 | 0.037533 ± 0.003445 | 0.036851 ± 0.003854 | 0.401545 ± 0.031280 |

NeuMF achieved the strongest mean full-candidate ranking performance
among the evaluated individual models and remained substantially more
stable than the validation-selected Hybrid.

Hybrid occasionally improved HR@10 for an individual seed, but did not
provide a robust ranking advantage. Its average nDCG@10 and MRR were
lower and its seed-to-seed variance was substantially larger.

This is an end-to-end pipeline robustness study, not a pure neural
initialization study: the random seed affects both stochastic model
training and the 800 validation users sampled for fusion-weight search.

## Sampled Ranking — 1 Positive + 99 Negatives

The sampled 1-positive + 99-negative protocol is retained as a
**secondary diagnostic**, not the primary model-selection protocol.

Across five seeds, NeuMF achieved:

- HR@10(sampled): 0.695464 ± 0.004796
- nDCG@10(sampled): 0.415613 ± 0.001321
- AUC(sampled): 0.905311 ± 0.000531

Sampled evaluation is much easier than ranking the held-out positive
against the full unseen catalogue and can change the apparent relative
strength of models.

Frozen V3 therefore uses **full-candidate ranking as the primary
benchmark**.

The historical AUC value 0.9127 is retired because it was affected by
the legacy candidate-set and popularity-scaling defects.

## Rating Prediction vs Ranking

| Model | MAE | RMSE |
|---|---:|---:|
| Global Mean | 0.9893 | 1.1756 |
| ItemKNN | 0.7382 | 0.9569 |
| Biased MF | 0.7175 | 0.9083 |

Biased MF is the strongest rating predictor but reaches only HR@10 = 0.0225
under full-candidate ranking.

This demonstrates that rating prediction accuracy and Top-N ranking quality
are different objectives.

## Fusion Weights

Fusion weights are selected only on validation data using a 0.1-step
grid search over 800 validation users.

| Seed | ItemKNN | Item-CF | MF | NeuMF | Content |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.2 | 0.1 | 0.1 | 0.5 | 0.1 |
| 43 | 0.2 | 0.2 | 0.2 | 0.1 | 0.3 |
| 44 | 0.1 | 0.2 | 0.2 | 0.4 | 0.1 |
| 45 | 0.0 | 0.1 | 0.1 | 0.6 | 0.2 |
| 46 | 0.0 | 0.2 | 0.0 | 0.7 | 0.1 |

The selected weights vary materially across seeds.

Paired Hybrid minus NeuMF differences across seeds 42-46:

| Metric | Mean Difference | SD |
|---|---:|---:|
| HR@10 | -0.000497 | 0.005520 |
| nDCG@10 | -0.001819 | 0.003553 |
| MRR | -0.002565 | 0.004022 |
| Coverage | -0.055061 | 0.029327 |

The evidence therefore does not justify claiming that additional fusion
complexity consistently improves ranking quality.

## Popularity / Exposure Trade-off

Frozen V3 keeps the bounded additive popularity penalty and evaluates
different alpha values within the same Hybrid system.

| 设置 | Precision@10 | Recall@10 | F1@10 | nDCG@10 | HR@10 | MRR | MedianRank | Coverage | CoveredItems | Diversity | Novelty | Gini | Top1%ItemShare | EvalUsers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 修正公式 alpha=0.0 | 0.0079 | 0.0791 | 0.0144 | 0.0383 | 0.0791 | 0.0378 | 157.0000 | 0.3724 | 1446.0000 | 0.7094 | 2.3532 | 0.9067 | 0.3011 | 6040.0000 |
| 修正公式 alpha=0.05 | 0.0080 | 0.0796 | 0.0145 | 0.0385 | 0.0796 | 0.0378 | 158.0000 | 0.3873 | 1504.0000 | 0.7053 | 2.3996 | 0.9010 | 0.2880 | 6040.0000 |
| 修正公式 alpha=0.1 | 0.0080 | 0.0796 | 0.0145 | 0.0382 | 0.0796 | 0.0374 | 157.0000 | 0.4028 | 1564.0000 | 0.7009 | 2.4493 | 0.8948 | 0.2747 | 6040.0000 |
| 修正公式 alpha=0.2 | 0.0078 | 0.0781 | 0.0142 | 0.0372 | 0.0781 | 0.0365 | 158.0000 | 0.4394 | 1706.0000 | 0.6922 | 2.5538 | 0.8813 | 0.2494 | 6040.0000 |
| 修正公式 alpha=0.4 | 0.0073 | 0.0732 | 0.0133 | 0.0344 | 0.0732 | 0.0339 | 182.0000 | 0.5241 | 2035.0000 | 0.6730 | 2.8215 | 0.8446 | 0.1971 | 6040.0000 |
| 原实现公式 score*(1-0.2*pop) | 0.0002 | 0.0022 | 0.0004 | 0.0014 | 0.0022 | 0.0019 | 3534.0000 | 0.3049 | 1184.0000 | 0.6653 | 3.1349 | 0.9246 | 0.3640 | 6040.0000 |

The default remains alpha = 0.10.

This ablation should be interpreted as an accuracy / exposure trade-off
within the same Hybrid model. It should not be used to infer that Hybrid
must have higher Coverage than NeuMF, because the two systems have
different score distributions and ranking policies.

Alpha was not retuned after inspecting Frozen V3 test results.

## Methodology Notes

### MF vs NeuMF: objective boundary

Biased MF and NeuMF should not be interpreted as a controlled architecture
comparison.

Biased MF is trained for explicit rating regression, while NeuMF is trained
for implicit-feedback ranking with negative sampling and Binary Cross-Entropy.
The observed gap therefore supports the conclusion that **rating prediction
and Top-N ranking are different objectives**, but it does not by itself prove
that the NeuMF architecture is superior to matrix factorization under the same
implicit-ranking objective.

A stronger architecture-level comparison would require an implicit MF
baseline such as BPR-MF or iALS.

### NeuMF training configuration

Frozen V3 uses:

- maximum 25 training epochs with validation-based early stopping;
- 4 sampled training negatives per positive interaction;
- GMF dimension: 32;
- MLP embedding dimension: 32;
- MLP layers: [64, 32, 16];
- end-to-end training from scratch;
- no separate GMF / MLP pretraining;
- validation HR@10 for early stopping.

The configuration was intended to provide a defensible neural-ranking
baseline rather than to reproduce or tune for state-of-the-art benchmark
performance.

### Content signal interpretation

The content recommender is weak as a standalone ranking model. Across the
five Frozen V3 seeds, its validation-selected fusion weight ranges from
0.1 to 0.3.

This should be interpreted only as a **validation-selected complementary
signal**. Without a leave-one-signal-out ablation, the experiment does not
establish that the content component independently improves held-out test
performance.

### Popularity penalty interpretation

The popularity-penalty ablation compares different alpha values **within the
same Hybrid system**.

For example, Coverage increases from 0.3724 at alpha=0 to 0.4028 at
alpha=0.10. This shows that the penalty reduces catalogue concentration inside
the Hybrid system.

It does not imply that Hybrid + penalty must achieve higher Coverage than
NeuMF, because NeuMF has a different score distribution and recommendation
policy.

### Statistical robustness

Frozen V3 reports an end-to-end robustness study across five fixed seeds:
42, 43, 44, 45 and 46.

NeuMF achieved:

- HR@10 = 0.078444 ± 0.001155
- nDCG@10 = 0.039352 ± 0.000641
- MRR = 0.039417 ± 0.000403
- Coverage = 0.456606 ± 0.011687

Hybrid achieved:

- HR@10 = 0.077947 ± 0.005715
- nDCG@10 = 0.037533 ± 0.003445
- MRR = 0.036851 ± 0.003854
- Coverage = 0.401545 ± 0.031280

The seed affects both stochastic model training and the 800 validation
users sampled for fusion-weight search. These results therefore measure
**pipeline-level robustness**, not pure neural-initialization variance.

The results are descriptive robustness evidence rather than a formal
statistical-significance test.

### Literature context

The project methodology is consistent with several established findings in
recommender-system research:

- He et al. (2017), *Neural Collaborative Filtering* — NeuMF and implicit
  negative-sampling formulation.
- Ferrari Dacrema et al. (2019), *Are We Really Making Much Progress?* —
  emphasizes the importance of strong, well-tuned classical baselines.
- Rendle et al. (2020), *Neural Collaborative Filtering vs. Matrix
  Factorization Revisited* — shows that simpler interaction models can remain
  highly competitive against neural similarity functions.
- Krichene & Rendle (2020), *On Sampled Metrics for Item Recommendation* —
  motivates treating sampled ranking metrics separately from full-catalogue
  evaluation.

These references provide methodological context rather than evidence that
Frozen V3 reproduces neither the exact datasets nor the exact experimental settings of those papers; the references provide methodological context only.

## Evaluation Audit

The project evolved from a recommender implementation into an
evaluation-audit project after several metrics behaved implausibly.

Three major correctness defects were identified:

1. **Candidate-set defect** — the held-out positive could be removed
   from ranking candidates, making Top-N hits structurally impossible.

2. **Popularity-scaling defect** — multiplying scores by raw popularity
   counts could invert score signs and destroy the ranking.

3. **Exact-score tie defect** — Frozen V2 ranked a positive by counting
   only candidates whose score was strictly greater than the positive
   score, assigning the positive the best possible position inside a tie.

The ItemKNN diagnostic found:

- 115 Top-10 hits
- 115 / 115 assigned rank 1
- 115 / 115 inside exact-score tie blocks
- 0 / 115 unique-score hits
- median tie-block size = 72
- maximum tie-block size = 666

Frozen V3 uses one deterministic ranking rule everywhere:

```text
1. score descending
2. internal item index ascending for exact ties
```

The same rule is used for Top-N generation, full-candidate ranking,
sampled ranking and NeuMF validation.

After correction, canonical ItemKNN became:

```text
HR@10   = 0.005629
nDCG@10 = 0.002730
```

Detailed evidence is preserved in:

`results/frozen_v3/tie_audit.txt`

## Tests

The repository contains **19 regression / invariant tests**.

```bash
python -m pytest -q
# 19 passed
```

Coverage includes:

- chronological and disjoint train / validation / test splits
- candidate-set correctness
- held-out positive availability
- information-boundary checks
- perfect / worst ranker sanity checks
- popularity-penalty behavior
- NeuMF negative sampling
- deterministic exact-score tie handling
- consistency between ranking metrics and Top-N ordering

The tests use synthetic data and do not require MovieLens-1M.

## Reproduce

Canonical Frozen V3 reproduction:

```bash
python run.py \
  --data-dir data/ml-1m \
  --out-dir outputs/reproduce_v3 \
  --seed 42 \
  --neumf-epochs 25 \
  --ablation \
  --bug-ablation
```

`NeuMFConfig.epochs` now defaults to 25, matching Frozen V3.

Fast smoke test:

```bash
python run.py --data-dir data/ml-1m --quick
```

SBERT is optional and was **not used** in Frozen V3.

## Evidence

The canonical evidence bundle is stored under:

`results/frozen_v3/`

It contains:

- `README.md`
- `metrics_seed42.json`
- `frozen_v3_multiseed_raw.csv`
- `frozen_v3_multiseed_summary.csv`
- `frozen_v3_fusion_weights.csv`
- `ablation_popularity.csv`
- `ablation_bugs.csv`
- `tie_audit.txt`
- `pytest.txt`
- `environment.txt`
- `SHA256SUMS.txt`

Frozen V2 remains in repository history for auditability, but its
ranking metrics are superseded by Frozen V3.

## Key Findings

- Evaluation correctness mattered more than adding model complexity:
  candidate construction, score scaling and tie handling each materially
  changed the conclusions.
- NeuMF was the strongest mean individual full-candidate ranker across
  five seeds.
- Item-CF remained a strong deterministic baseline.
- Validation-selected Hybrid did not show a robust advantage over
  NeuMF and introduced substantially higher pipeline-level variance.
- Sampled and full-candidate evaluation are not interchangeable.
- Rating prediction and Top-N ranking optimize different objectives.
- Popularity control is an accuracy / exposure trade-off rather than a
  free performance improvement.

## Limitations

This is an offline benchmark rather than a production recommendation service.

It does not include online A/B testing, real-time serving infrastructure, or
production user-impact evidence.

