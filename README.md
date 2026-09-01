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
any frozen run.

## Frozen V4 — Full-Candidate Ranking

**Frozen V4 is the current revised engineering benchmark.** Because this methodology revision followed inspection of earlier test results, it should not be interpreted as a fresh unbiased holdout.

Earlier frozen sets are superseded:

- Frozen V2 ranking metrics were affected by optimistic exact-score
  tie handling (see *Evaluation Audit*, defect 3).
- Frozen V3 fixed tie handling but selected fusion weights on an
  800-user validation subsample, which made the selection step itself
  overfit (see *Evaluation Audit*, defect 4).

Frozen V4 protocol:

- MovieLens-1M
- chronological train / validation / test holdout
- canonical seed = 42
- NeuMF maximum epochs = 25
- fusion weights searched on **all 6,040 validation users**
- device = CPU (see *Reproducibility*)
- full-candidate Top-10 ranking as the primary protocol
- deterministic tie-breaking:
  score descending, then internal item index ascending

### Five-seed pipeline robustness

Seeds: 42, 43, 44, 45, 46.

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

Under the Mac CPU Frozen V4 run, Hybrid has the highest mean HR@10,
nDCG@10 and MRR among the evaluated systems. Its seed-to-seed variance
is also lower than NeuMF's on these ranking metrics.

Paired per-seed comparison against the strongest single model of the
same seed:

| Seed | Best single | Best single HR@10 | Hybrid HR@10 | Δ HR@10 | Δ nDCG@10 |
|---:|---|---:|---:|---:|---:|
| 42 | NeuMF | 0.0791 | 0.0811 | +0.0020 | -0.0003 |
| 43 | NeuMF | 0.0772 | 0.0826 | +0.0055 | +0.0030 |
| 44 | Item-CF (implicit) | 0.0762 | 0.0834 | +0.0073 | +0.0021 |
| 45 | Item-CF (implicit) | 0.0762 | 0.0848 | +0.0086 | +0.0032 |
| 46 | NeuMF | 0.0773 | 0.0836 | +0.0063 | +0.0012 |

Mean paired Δ HR@10 = **+0.005927 ± 0.002496**,
with Hybrid ahead in **5/5 seeds**.

Mean paired Δ nDCG@10 = **+0.001859 ± 0.001415**,
with Hybrid ahead in **4/5 seeds**.

Relative to the strongest single model within each seed, the mean lift is
approximately **7.68% in HR@10** and
**4.85% in nDCG@10**.

Frozen V4 is a revised engineering benchmark rather than a fresh,
previously untouched holdout. The methodology revision followed
inspection of earlier test results, so these numbers are not presented
as a new unbiased model-selection experiment.

## Sampled Ranking — 1 Positive + 99 Negatives

The sampled 1-positive + 99-negative protocol is retained as a
**secondary diagnostic**, not the primary model-selection protocol.

Five-seed means:

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

Sampled evaluation is substantially easier than ranking the held-out
positive against the full unseen catalogue, and the two protocols can
reorder systems.

Hybrid leads sampled HR@10 and nDCG@10, while NeuMF leads sampled AUC.

Frozen V4 therefore keeps **full-candidate ranking as the primary
benchmark** and reports sampled evaluation only as a secondary
diagnostic and for comparability with prior recommender literature.

The historical AUC value 0.9127 remains retired because it was affected
by the legacy candidate-set and popularity-scaling defects.

## Rating Prediction vs Ranking

| Model | MAE | RMSE |
|---|---:|---:|
| Global Mean | 0.9893 | 1.1756 |
| ItemKNN | 0.7382 | 0.9569 |
| Biased MF | 0.7175 | 0.9083 |

Biased MF is the strongest rating predictor but reaches only HR@10 = 0.0222 under the seed-42 full-candidate ranking.

In this benchmark, rating-prediction accuracy and Top-N ranking utility are different objectives.

## Fusion Weights

Fusion weights are selected on validation data only, using a 0.1-step
grid search over all **6,040 validation users**. The test split is not
used by the search procedure itself.

| Seed | ItemKNN | Item-CF | MF | NeuMF | Content |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.0 | 0.1 | 0.1 | 0.6 | 0.2 |
| 43 | 0.0 | 0.2 | 0.0 | 0.7 | 0.1 |
| 44 | 0.0 | 0.2 | 0.0 | 0.6 | 0.2 |
| 45 | 0.0 | 0.2 | 0.0 | 0.6 | 0.2 |
| 46 | 0.0 | 0.2 | 0.0 | 0.6 | 0.2 |

The Mac Frozen V4 solutions are concentrated around NeuMF, Item-CF and
Content. NeuMF carries **0.6–0.7** weight in every seed.

ItemKNN receives **zero weight in all five seeds**. Biased MF receives
zero weight in four seeds and 0.1 in seed 42.

This is benchmark-specific evidence that strong explicit-rating
prediction does not automatically provide incremental Top-N ranking
utility. It does **not** imply that rating-based models are universally
useless in ranking ensembles.

## Popularity / Exposure Trade-off

Frozen V4 keeps the bounded additive popularity penalty. The following
ablation is read directly from the **Mac canonical seed-42 artifact**:

| Setting | HR@10 | nDCG@10 | Coverage | Diversity | Novelty | Gini |
|---|---:|---:|---:|---:|---:|---:|
| alpha=0.0 | 0.0808 | 0.0394 | 0.4066 | 0.6894 | 2.4465 | 0.8923 |
| alpha=0.05 | 0.0808 | 0.0391 | 0.4218 | 0.6852 | 2.4940 | 0.8860 |
| alpha=0.1 | 0.0811 | 0.0392 | 0.4399 | 0.6804 | 2.5469 | 0.8789 |
| alpha=0.2 | 0.0805 | 0.0384 | 0.4759 | 0.6718 | 2.6557 | 0.8637 |
| alpha=0.4 | 0.0760 | 0.0362 | 0.5555 | 0.6524 | 2.9159 | 0.8254 |
| 原实现公式 score*(1-0.2*pop) | 0.0023 | 0.0011 | 0.3178 | 0.6114 | 2.9642 | 0.9098 |

The popularity penalty represents a ranking-versus-exposure trade-off,
not a free improvement.

Alpha = 0.10 remains the pre-existing default and was **not retuned
after inspecting Frozen V4 test results**.

The legacy multiplicative formula is retained only as a controlled
failure reproduction because it can invert score signs; it is not a
valid tuning point.

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

Four major correctness defects were identified. The first three are
scoring/ranking bugs; the fourth is a defect in the *model-selection
procedure* rather than in any model.

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

Detailed evidence is preserved in `results/frozen_v3/tie_audit.txt`.

### Defect 4 — selection variance in fusion-weight search

Frozen V3 searched 1,001 fusion-weight combinations using a randomly
selected 800-user validation subset and full-candidate nDCG@10.

Because validation nDCG@10 is only around 0.04 under the full-candidate
protocol, the 800-user subset contains relatively few Top-10 hits. This
makes the selected optimum sensitive to which validation users happen
to be sampled.

A controlled diagnostic isolates **one source of this instability**:
the seed-42 model score matrices are trained once and then held fixed,
while only the validation-user subset is redrawn.

Across repeated redraws, the selected NeuMF weight ranged from 0.2 to
1.0 even though no model parameter changed.

This establishes the narrower claim that **validation-user subsampling
alone is sufficient to induce substantial fusion-selection variance**.
It does not establish that all Frozen V3 pipeline variance came from
that factor.

The same fixed score matrices were also used to compare three search
settings:

| Search configuration | Observed test HR@10 |
|---|---:|
| 800 users, full-candidate objective | 0.0777 ± 0.0026 |
| all 6,040 users, full-candidate objective | 0.0810 |
| all 6,040 users, sampled-negative objective | 0.0823 |

These A/B/C comparisons are **exploratory / post-hoc diagnostics**
because the same test split had already been inspected. They motivated
the engineering revision but are not treated as a fresh unbiased
holdout.

Frozen V4 therefore defaults to all validation users and keeps
`search_objective = full`, so the selection objective matches the
primary reported ranking metric.

Reproduce the diagnostic with:

    python -m experiments.weight_search_stability --data-dir data/ml-1m

## Tests

The repository contains **25 regression / invariant tests**.

```bash
python -m pytest -q
# 25 passed
```

Coverage includes:

- chronological and disjoint train / validation / test splits
- candidate-set correctness and held-out positive availability
- information-boundary checks
- perfect / worst-ranker sanity checks
- popularity-penalty behavior
- deterministic exact-score tie handling
- consistency between ranking metrics and Top-N ordering
- bounded NeuMF negative sampling with exact-complement fallback
- explicit failure when no legal negative candidate exists
- prevention of positive-target padding in sampled fusion search
- agreement between sampled-search and global tie policies
- `search_users = 0` resolving to all validation users
- Frozen evidence-bundle integrity guards

The unit and invariant tests use synthetic data and do not require
MovieLens-1M.

## Reproduce

Frozen V4 reproduction (five seeds, ~70 min on 2 CPU cores):

```bash
python -m experiments.multiseed \
  --data-dir data/ml-1m \
  --seeds 42 43 44 45 46 \
  --out-dir results/frozen_v4

python -m experiments.make_frozen_bundle \
  --run-dir results/frozen_v4 --version v4
```

Single canonical run:

```bash
python run.py \
  --data-dir data/ml-1m \
  --out-dir outputs/reproduce_v4 \
  --seed 42 \
  --neumf-epochs 25 \
  --ablation \
  --bug-ablation
```

Fast smoke test:

```bash
python run.py --data-dir data/ml-1m --quick
```

SBERT is optional and was **not used** in Frozen V4.

### Reproducibility

All deterministic components reproduce **bit-for-bit across platforms**.
Verified between macOS 26 / arm64 / Python 3.14 / NumPy 2.5 and
Linux x86-64 / Python 3.11 / NumPy 2.4: ItemKNN, Item-CF, Biased MF,
Content and Popularity returned identical HR@10 and nDCG@10 to every
printed digit.

NeuMF does **not** reproduce bit-for-bit across platforms. Floating-point
accumulation order differs between BLAS builds, which shifts the
per-epoch validation HR and therefore the early-stopping point:

| Platform | Best epoch | Stopped at | Test HR@10 |
|---|---:|---:|---:|
| macOS arm64, Python 3.14 | 17 | 19 | 0.0795 |
| Linux x86-64, Python 3.11 | 13 | 15 | 0.0775 |

The spread is about ±0.002 HR@10. Frozen runs therefore record the
realised epochs in `training_details.csv` and in the `training` block of
`metrics_seed42.json`, so any deviation can be traced rather than guessed.

For this reason `--device cpu` is the default. `--device auto` enables
MPS on Apple Silicon or CUDA where available and is considerably faster,
but its numbers are not comparable to the frozen benchmark.

## Evidence

The canonical evidence bundle is `results/frozen_v4/`:

- `README.md`
- `metrics_seed42.json`
- `recommendations_sample_seed42.csv`
- `multiseed_raw.csv`
- `multiseed_summary.csv`
- `fusion_weights.csv`
- `training_details.csv`
- `paired_hybrid_vs_best_single.csv`
- `ablation_popularity.csv`
- `ablation_bugs.csv`
- `environment.txt`
- `pytest.txt`
- `SHA256SUMS.txt`

Verify integrity with:

```bash
cd results/frozen_v4 && sha256sum -c SHA256SUMS.txt
```

The bundle and its checksum manifest are generated by
`experiments/make_frozen_bundle.py`, which also fails if any listed file
would be excluded by `.gitignore` — a check added after Frozen V3
shipped a manifest referencing an ignored file.

`results/frozen_v3/` is retained for auditability. Its ranking metrics
are correct under the tie policy but its fusion weights and Hybrid
results are superseded, for the reason given in *Defect 4*.

## Key Findings

- **Evaluation correctness materially changed the benchmark conclusion.**
  Candidate construction, popularity scaling and exact-score tie handling
  all required explicit auditing.
- **Model selection itself also required auditing.** With model scores held
  fixed, validation-user subsampling alone was sufficient to create
  substantial fusion-weight variance.
- Under the Mac CPU Frozen V4 revised protocol, Hybrid reached
  **HR@10 = 0.083113 ± 0.001350**
  and
  **nDCG@10 = 0.040153 ± 0.000696**.
- Hybrid beat the strongest single model on HR@10 in **5/5 seeds**
  and on nDCG@10 in **4/5 seeds**.
- Mean paired HR@10 improvement was **+0.005927**.
- ItemKNN received zero fusion weight in all five seeds; Biased MF received
  zero in four of five.
- Strong rating RMSE therefore did not automatically translate into
  incremental ranking utility in this benchmark.
- Sampled and full-candidate evaluation are not interchangeable and can
  reorder systems.
- Popularity control remains an accuracy / exposure trade-off.
- Frozen V4 is a **revised engineering benchmark**, not a newly untouched
  statistical holdout, because the methodology was revised after earlier
  test results had already been inspected.

## Limitations

This is an offline benchmark rather than a production recommendation service.

It does not include online A/B testing, real-time serving infrastructure, or
production user-impact evidence.

