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

## Frozen V2 — Full-Candidate Ranking

| Model | HR@10 | nDCG@10 | Coverage |
|---|---:|---:|---:|
| Random | 0.0023 | 0.0010 | 1.0000 |
| Popularity | 0.0369 | 0.0180 | 0.0512 |
| ItemKNN | 0.0190 | 0.0190 | 0.6348 |
| Item-CF | 0.0762 | 0.0378 | 0.3235 |
| Biased MF | 0.0225 | 0.0118 | 0.2859 |
| NeuMF | 0.0795 | 0.0398 | 0.4757 |
| Content | 0.0119 | 0.0059 | 0.5735 |
| Hybrid | 0.0796 | 0.0382 | 0.4028 |

NeuMF is the strongest individual ranking model.

Hybrid reaches essentially the same HR@10 but does not improve nDCG over
NeuMF, so the held-out evidence does not justify claiming that additional
fusion complexity consistently improves ranking quality.

## Sampled Ranking — 1 Positive + 99 Negatives

| Model | HR@10 | nDCG@10 | AUC |
|---|---:|---:|---:|
| Random | 0.1013 | 0.0438 | 0.5061 |
| Popularity | 0.4733 | 0.2602 | 0.8221 |
| ItemKNN | 0.2248 | 0.1048 | 0.6292 |
| Item-CF | 0.6684 | 0.3913 | 0.8909 |
| Biased MF | 0.2748 | 0.1485 | 0.6255 |
| NeuMF | 0.6917 | 0.4147 | 0.9046 |
| Content | 0.1980 | 0.1027 | 0.5280 |
| Hybrid | 0.6873 | 0.4136 | 0.8609 |

Sampled metrics are reported separately because the sampled protocol is
substantially easier than full-candidate ranking.

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

| Signal | Weight |
|---|---:|
| ItemKNN | 0.20 |
| Item-CF | 0.10 |
| Biased MF | 0.10 |
| NeuMF | 0.50 |
| Content | 0.10 |

Weights were selected using validation nDCG@10.

Test metrics were not used for fusion-weight selection.

## Popularity / Exposure Trade-off

| Alpha | HR@10 | nDCG@10 | Coverage | Novelty | Gini |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0791 | 0.0383 | 0.3724 | 2.3532 | 0.9067 |
| 0.05 | 0.0796 | 0.0385 | 0.3873 | 2.3996 | 0.9010 |
| 0.10 | 0.0796 | 0.0382 | 0.4028 | 2.4493 | 0.8948 |
| 0.20 | 0.0781 | 0.0372 | 0.4394 | 2.5538 | 0.8813 |
| 0.40 | 0.0732 | 0.0344 | 0.5241 | 2.8215 | 0.8446 |

Frozen V2 uses alpha = 0.10.

Compared with alpha = 0, catalogue coverage increases from 37.24% to
40.28% while HR@10 remains essentially unchanged.

No parameter was changed after inspecting Frozen V2 test results.

## Evaluation Audit

Controlled historical evaluation failures are isolated in:

`experiments/legacy_bug_ablation.py`

The experiment reproduces candidate-set and popularity-scaling failure modes
without mixing them into the production recommendation pipeline.

## Tests

The repository contains 17 regression and invariant tests covering temporal
splitting, leakage prevention, candidate construction, NeuMF negative
sampling, metric behavior, and popularity controls.

Run:

    python -m pytest -q

Frozen V2 result:

    17 passed

## Reproduce

Install dependencies:

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt

Run the canonical benchmark:

    python run.py --data-dir data/ml-1m --out-dir outputs/reproduce_v2 --seed 42 --top-n 10 --pop-alpha 0.10 --normalize zscore --mf-epochs 40 --neumf-epochs 25 --ablation --bug-ablation

Reference CPU runtime: approximately 2.8 minutes on Apple M5.

## Evidence

Canonical frozen evidence is stored in:

`results/frozen_v2/`

## Key Findings

1. Evaluation correctness comes before model comparison.
2. Rating prediction and Top-N ranking are different objectives.
3. NeuMF is the strongest individual ranking model.
4. Hybrid fusion does not consistently outperform NeuMF.
5. Mild popularity control improves catalogue exposure with essentially no hit-rate loss.
6. Sampled ranking is materially easier than full-candidate ranking.
7. Additional complexity should be justified by held-out evidence.

## Limitations

This is an offline benchmark rather than a production recommendation service.

It does not include online A/B testing, real-time serving infrastructure, or
production user-impact evidence.

