# Frozen V2 Experimental Results

## Protocol

- MovieLens-1M
- 1,000,209 ratings
- 6,040 users
- 3,883 movies
- Train: 988,129
- Validation: 6,040
- Test: 6,040
- Seed: 42
- Top-N: 10
- Primary protocol: full-candidate ranking
- Secondary protocol: 1 positive + 99 negatives
- Fusion selection: validation nDCG@10

## Full-Candidate Ranking

| Model | HR@10 | nDCG@10 | MRR | Coverage |
|---|---:|---:|---:|---:|
| Random | 0.0023 | 0.0010 | 0.0021 | 1.0000 |
| Popularity | 0.0369 | 0.0180 | 0.0199 | 0.0512 |
| ItemKNN | 0.0190 | 0.0190 | 0.0213 | 0.6348 |
| Item-CF | 0.0762 | 0.0378 | 0.0372 | 0.3235 |
| Biased MF | 0.0225 | 0.0118 | 0.0127 | 0.2859 |
| NeuMF | 0.0795 | 0.0398 | 0.0396 | 0.4757 |
| Content | 0.0119 | 0.0059 | 0.0073 | 0.5735 |
| Hybrid | 0.0796 | 0.0382 | 0.0374 | 0.4028 |

## Sampled Ranking

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

## Rating Prediction

| Model | MAE | RMSE |
|---|---:|---:|
| Global Mean | 0.9893 | 1.1756 |
| ItemKNN | 0.7382 | 0.9569 |
| Biased MF | 0.7175 | 0.9083 |

## Fusion Weights

| Signal | Weight |
|---|---:|
| ItemKNN | 0.20 |
| Item-CF | 0.10 |
| Biased MF | 0.10 |
| NeuMF | 0.50 |
| Content | 0.10 |

## Popularity Ablation

| Alpha | HR@10 | nDCG@10 | Coverage | Novelty | Gini |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0791 | 0.0383 | 0.3724 | 2.3532 | 0.9067 |
| 0.05 | 0.0796 | 0.0385 | 0.3873 | 2.3996 | 0.9010 |
| 0.10 | 0.0796 | 0.0382 | 0.4028 | 2.4493 | 0.8948 |
| 0.20 | 0.0781 | 0.0372 | 0.4394 | 2.5538 | 0.8813 |
| 0.40 | 0.0732 | 0.0344 | 0.5241 | 2.8215 | 0.8446 |

## Runtime

- MF early stop: epoch 11
- NeuMF early stop: epoch 19
- Fusion search: 1,001 combinations over 800 validation users
- Reference CPU runtime: approximately 2.8 minutes

## Tests

    17 passed

## Final Conclusions

1. Evaluation correctness must be established before model comparison.
2. Rating prediction and Top-N ranking measure different objectives.
3. NeuMF is the strongest individual ranking model.
4. Hybrid fusion does not consistently outperform NeuMF.
5. Popularity control improves catalogue exposure with minimal hit-rate loss.
6. Sampled ranking is substantially easier than full-candidate ranking.

