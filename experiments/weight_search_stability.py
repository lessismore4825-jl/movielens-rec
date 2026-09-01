"""诊断：融合权重搜索的不稳定性来自验证集子采样，还是来自模型训练随机性？

Frozen V3 的五种子实验里，搜出的 NeuMF 权重从 0.1 跳到 0.7，Hybrid 的
HR@10 标准差(0.0057)是 NeuMF(0.0012)的五倍。但那个实验同时变动了两件事：

  1. 模型训练的随机性（NeuMF/MF 的初始化与负采样）
  2. 权重搜索所用的 800 位验证用户子样本

因此无法判断不稳定性来自哪一侧。本脚本把模型固定（只训练一次），
只变动第 2 项，从而把两者分离。

同时对比三种搜索设置：
  A. 800 位验证用户 + 全候选 nDCG@10 目标（Frozen V3 现状）
  B. 全部 6040 位验证用户 + 全候选 nDCG@10 目标
  C. 全部验证用户 + 采样负例(1正+99负) nDCG@10 目标

动机：全候选协议下验证 nDCG@10 约 0.04，800 位用户里只有约 48 次命中，
信噪比极低；而采样负例协议下 HR@10 约 0.7，同样的用户数能提供多一个
数量级的有效信号。若 A 的方差显著大于 B/C，则说明验证用户子采样本身就足以产生
显著的融合权重选择方差。该实验不用于证明训练随机性没有贡献。

用法：
    python -m experiments.weight_search_stability --data-dir data/ml-1m
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import Config, PathConfig, set_seed
from recsys.evaluate import positive_rank
from recsys.fusion import HybridFusion, normalize_scores
from recsys.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODELS = ["itemknn", "itemcf", "mf", "neumf", "content"]


def ndcg_at_n(scores: np.ndarray, targets: np.ndarray, n: int = 10) -> float:
    rank = positive_rank(scores, targets)
    return float(np.where(rank < n, 1.0 / np.log2(rank + 2.0), 0.0).mean())


def build_grid(step: float = 0.1, k: int = 5) -> list[tuple]:
    g = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    return [c for c in itertools.product(g, repeat=k) if abs(sum(c) - 1.0) < 1e-6]


def search(norm_cache: dict[str, np.ndarray], mask: np.ndarray, targets: np.ndarray,
           pop: np.ndarray, alpha: float, grid: list[tuple],
           sampled_negatives: np.ndarray | None = None) -> tuple[dict, float]:
    """在给定用户子集上网格搜索权重，返回 (最优权重, 验证目标值)。"""
    best_w, best_v = None, -np.inf
    n = len(targets)
    rows = np.arange(n)

    for combo in grid:
        total = np.zeros_like(norm_cache[MODELS[0]])
        for name, w in zip(MODELS, combo):
            if w:
                total += np.float32(w) * norm_cache[name]
        if alpha > 0:
            lo = np.where(mask, total, np.inf).min(axis=1, keepdims=True)
            hi = np.where(mask, total, -np.inf).max(axis=1, keepdims=True)
            total = (total - lo) / np.maximum(hi - lo, 1e-8)
            total = total - np.float32(alpha) * pop[None, :]
        total = np.where(mask, total, -np.inf)

        if sampled_negatives is None:
            v = ndcg_at_n(total, targets)
        else:
            # 采样负例协议：只在 1 正 + 99 负的小候选集上算 nDCG，信号密度高得多
            cand = np.concatenate([targets[:, None], sampled_negatives], axis=1)
            s = np.take_along_axis(total, cand, axis=1)
            neg_better = ((s[:, 1:] > s[:, :1])
                          | ((s[:, 1:] == s[:, :1]) & (cand[:, 1:] < targets[:, None])))
            rank = neg_better.sum(axis=1)
            v = float(np.where(rank < 10, 1.0 / np.log2(rank + 2.0), 0.0).mean())

        if v > best_v:
            best_v, best_w = v, dict(zip(MODELS, combo))
    return best_w, best_v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/weight_stability"))
    p.add_argument("--trials", type=int, default=20, help="800 用户子采样的重复次数")
    p.add_argument("--subsample", type=int, default=800)
    p.add_argument("--neumf-epochs", type=int, default=25)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(paths=PathConfig(data_dir=a.data_dir, out_dir=a.out_dir), seed=42)
    cfg.neumf.epochs = a.neumf_epochs
    cfg.fusion.search_weights = False          # 搜索由本脚本自己做
    set_seed(cfg.seed)

    # ---- 只训练一次，之后所有对比共用同一批打分矩阵 ----
    cache = a.out_dir / "scores_seed42.npz"
    pipe = Pipeline(cfg)
    ds = pipe.load()
    pipe.build_models()
    if cache.exists():
        logger.info("复用已缓存的打分矩阵 %s", cache)
        z = np.load(cache)
        pipe.scores = {k: z[k] for k in z.files}
    else:
        pipe.train_and_score()
        np.savez_compressed(cache, **pipe.scores)
        logger.info("打分矩阵已缓存到 %s", cache)

    scores = {m: pipe.scores[m] for m in MODELS}
    alpha = cfg.fusion.popularity_alpha
    pop = HybridFusion.popularity_norm(ds)
    grid = build_grid(cfg.fusion.search_step, len(MODELS))
    logger.info("权重网格共 %d 组合", len(grid))

    valid_mask = HybridFusion.candidate_mask(ds, target="valid")
    test_mask = HybridFusion.candidate_mask(ds, target="test")
    v_u = ds.valid["u"].to_numpy()
    v_i = ds.valid["i"].to_numpy()
    t_i = ds.test["i"].to_numpy()

    norm_valid_full = {m: normalize_scores(scores[m], valid_mask, cfg.fusion.normalize)
                       for m in MODELS}
    norm_test = {m: normalize_scores(scores[m], test_mask, cfg.fusion.normalize)
                 for m in MODELS}

    def test_metrics(w: dict) -> dict:
        total = np.zeros((ds.n_users, ds.n_items), dtype=np.float32)
        for m, wi in w.items():
            if wi:
                total += np.float32(wi) * norm_test[m]
        if alpha > 0:
            lo = np.where(test_mask, total, np.inf).min(axis=1, keepdims=True)
            hi = np.where(test_mask, total, -np.inf).max(axis=1, keepdims=True)
            total = (total - lo) / np.maximum(hi - lo, 1e-8)
            total = total - np.float32(alpha) * pop[None, :]
        total = np.where(test_mask, total, -np.inf)
        rank = positive_rank(total, t_i)
        return {"HR@10": float((rank < 10).mean()),
                "nDCG@10": float(np.where(rank < 10, 1.0 / np.log2(rank + 2.0), 0.0).mean())}

    rows = []

    # ---- A. 现状：800 位验证用户 + 全候选目标，重复 trials 次 ----
    for t in range(a.trials):
        rng = np.random.default_rng(1000 + t)
        sel = rng.choice(len(v_u), size=a.subsample, replace=False)
        su, si = v_u[sel], v_i[sel]
        sub_mask = valid_mask[su]
        cache_sub = {m: norm_valid_full[m][su] for m in MODELS}
        w, v = search(cache_sub, sub_mask, si, pop, alpha, grid)
        rows.append({"设置": f"A. {a.subsample}用户/全候选目标", "trial": t,
                     **w, "验证目标": v, **test_metrics(w)})
        logger.info("A trial %2d  weights=%s  test_HR=%.4f", t,
                    [w[m] for m in MODELS], rows[-1]["HR@10"])

    # ---- B. 全部验证用户 + 全候选目标 ----
    w, v = search(norm_valid_full, valid_mask, v_i, pop, alpha, grid)
    rows.append({"设置": "B. 全部6040用户/全候选目标", "trial": 0, **w,
                 "验证目标": v, **test_metrics(w)})
    logger.info("B  weights=%s  test_HR=%.4f", [w[m] for m in MODELS], rows[-1]["HR@10"])

    # ---- C. 全部验证用户 + 采样负例目标（信号更密） ----
    rng = np.random.default_rng(cfg.seed)
    n_neg = 99
    negs = np.empty((len(v_u), n_neg), dtype=np.int64)
    for r, uu in enumerate(v_u):
        pool = np.flatnonzero(valid_mask[r])
        pool = pool[pool != v_i[r]]
        negs[r] = rng.choice(pool, size=n_neg, replace=False)
    w, v = search(norm_valid_full, valid_mask, v_i, pop, alpha, grid, sampled_negatives=negs)
    rows.append({"设置": "C. 全部6040用户/采样负例目标", "trial": 0, **w,
                 "验证目标": v, **test_metrics(w)})
    logger.info("C  weights=%s  test_HR=%.4f", [w[m] for m in MODELS], rows[-1]["HR@10"])

    # ---- 单模型参照 ----
    for m in MODELS:
        rows.append({"设置": f"参照: 仅 {m}", "trial": 0,
                     **{k: (1.0 if k == m else 0.0) for k in MODELS},
                     "验证目标": np.nan,
                     **test_metrics({k: (1.0 if k == m else 0.0) for k in MODELS})})

    df = pd.DataFrame(rows)
    out = a.out_dir / "weight_search_stability.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n=== A 设置（800 用户子采样）在 %d 次重复下的表现 ===" % a.trials)
    sub = df[df["设置"].str.startswith("A.")]
    print(sub[MODELS + ["HR@10", "nDCG@10"]].describe().loc[["mean", "std", "min", "max"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== 各设置对比 ===")
    show = df[~df["设置"].str.startswith("A.")]
    agg = sub[["HR@10", "nDCG@10"]].agg(["mean", "std"])
    print(f"A. {a.subsample}用户/全候选目标      HR@10={agg.loc['mean','HR@10']:.4f} "
          f"± {agg.loc['std','HR@10']:.4f}   nDCG@10={agg.loc['mean','nDCG@10']:.4f} "
          f"± {agg.loc['std','nDCG@10']:.4f}")
    for _, r in show.iterrows():
        print(f"{r['设置']:<32} HR@10={r['HR@10']:.4f}              "
              f"nDCG@10={r['nDCG@10']:.4f}")
    print(f"\n明细已保存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
