"""复现原实现的两处 bug，在同一份数据切分上做受控对照。

目的不是"批评旧代码"，而是把"指标为 0"这件事的归因坐实：
报告原先把 Precision/Recall/F1/nDCG/HR 全为 0 归因于硬件没有 GPU、算力不足。
如果这个归因成立，那么在同样的算力下关掉这两个 bug，指标应当仍然是 0。
下面的对照实验表明：只要关掉 bug，指标立刻变成正数；只要打开 bug，
无论跑多久、用什么硬件，五项指标恒等于 0。这是一个可证伪的判定。

两处 bug
--------
A. 热度惩罚量纲错误
       score *= (1 - 0.2 * popularity_count)
   popularity_count ∈ [0, 3388]，乘数在 pop=5 处穿过 0 变负，排序被整体翻转。

B. 候选集把测试正例排除掉
       候选集 = 全部物品 − 用户在**全量 ratings** 中评过的物品
   测试正例来自 ratings.groupby('userId').tail(1)，本身就在全量 ratings 里，
   于是必然被当作"已看过"剔除，`full_scores.get(pos_item, 0)` 取默认值 0。
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from recsys import evaluate as ev
from recsys.data import Dataset
from recsys.fusion import HybridFusion, normalize_scores

logger = logging.getLogger(__name__)

# 原实现的权重与模型对应关系（cf→itemknn, svd→mf, ncf→neumf, content→content）
LEGACY_WEIGHTS = {"itemknn": 0.1, "mf": 0.4, "neumf": 0.4, "content": 0.1}


def _legacy_candidate_mask(ds: Dataset) -> np.ndarray:
    """bug B：用全量交互（train+valid+test）生成候选集，测试正例被排除。"""
    mask = np.ones((ds.n_users, ds.n_items), dtype=bool)
    for split in (ds.train, ds.valid, ds.test):
        mask[split["u"].to_numpy(), split["i"].to_numpy()] = False
    return mask


def _fused(scores: Dict[str, np.ndarray], mask: np.ndarray,
           weights: Dict[str, float], normalize: str = "minmax") -> np.ndarray:
    total = np.zeros(next(iter(scores.values())).shape, dtype=np.float32)
    for name, w in weights.items():
        if w and name in scores:
            total += np.float32(w) * normalize_scores(scores[name], mask, normalize)
    return total


def _metrics_legacy_convention(fused: np.ndarray, mask: np.ndarray, pos: np.ndarray,
                               ds: Dataset, top_n: int, rng: np.random.Generator,
                               n_neg: int = 100) -> Dict[str, float]:
    """按原实现的取分约定计算指标：正例若不在候选集内，其分数取默认值 0。"""
    n = fused.shape[0]
    scored = np.where(mask, fused, -np.inf)

    pos_in = mask[np.arange(n), pos]
    pos_score = np.where(pos_in, fused[np.arange(n), pos], 0.0)   # ← 关键：默认 0

    rank = (scored > pos_score[:, None]).sum(axis=1)
    hit = ((rank < top_n) & pos_in).astype(np.float64)
    ndcg = np.where(hit > 0, 1.0 / np.log2(rank + 2.0), 0.0)

    # 原实现的 AUC：从候选集中采负例，与正例分数比较
    aucs = np.empty(n)
    for r in range(n):
        pool = np.flatnonzero(mask[r])
        take = min(n_neg, len(pool))
        negs = fused[r, rng.choice(pool, size=take, replace=False)]
        d = pos_score[r] - negs
        aucs[r] = ((d > 0).sum() + 0.5 * (d == 0).sum()) / take

    rec = ev.top_n_items(scored, top_n)
    counts = np.bincount(rec.ravel(), minlength=ds.n_items)
    return {
        f"Precision@{top_n}": float((hit / top_n).mean()),
        f"Recall@{top_n}": float(hit.mean()),
        f"F1@{top_n}": float((2 * (hit / top_n) * hit /
                              np.maximum(hit / top_n + hit, 1e-12)).mean()),
        f"nDCG@{top_n}": float(ndcg.mean()),
        f"HR@{top_n}": float(hit.mean()),
        "AUC": float(aucs.mean()),
        "Coverage": float((counts > 0).sum() / ds.n_items),
        "CoveredItems": float((counts > 0).sum()),
        "Novelty": ev.novelty(rec, ds.item_popularity, ds.n_users),
        "正例在候选集内比例": float(pos_in.mean()),
    }


def run_bug_ablation(scores: Dict[str, np.ndarray], ds: Dataset,
                     top_n: int = 10, seed: int = 42) -> pd.DataFrame:
    """2×2 受控对照：{热度惩罚 bug 开/关} × {候选集 bug 开/关}。"""
    pop_raw = ds.item_popularity.astype(np.float32)
    clean_mask = HybridFusion.candidate_mask(ds, target="test")
    buggy_mask = _legacy_candidate_mask(ds)
    pos = ds.test["i"].to_numpy()

    rows = []
    for cand_bug in (False, True):
        mask = buggy_mask if cand_bug else clean_mask
        base = _fused(scores, mask, LEGACY_WEIGHTS)
        for pop_bug in (False, True):
            f = base * (1.0 - 0.2 * pop_raw[None, :]) if pop_bug else base
            m = _metrics_legacy_convention(
                f, mask, pos, ds, top_n, np.random.default_rng(seed))
            rows.append({
                "热度惩罚 bug": "开" if pop_bug else "关",
                "候选集 bug": "开" if cand_bug else "关",
                "对应版本": ("原实现" if (pop_bug and cand_bug)
                             else "本实现" if not (pop_bug or cand_bug) else "单 bug 对照"),
                **m,
            })
            del f
        del base

    return pd.DataFrame(rows)
