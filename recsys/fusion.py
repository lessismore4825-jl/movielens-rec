"""混合融合层：分数归一化、加权求和、热度惩罚与权重搜索。

修复的两个核心缺陷
------------------
**1. 热度惩罚的量纲错误（原项目所有异常现象的总根源）**

原实现：

    score *= (1 - 0.2 * self.movie_popularity.get(mid, 0))

`movie_popularity` 是原始交互次数，在 ml-1m 上取值 0 ~ 3388。于是：

    交互 0 次   -> 乘数 = 1 - 0     =    1.0     分数不变
    交互 100 次 -> 乘数 = 1 - 20    =  -19.0     分数变号
    交互 3000 次-> 乘数 = 1 - 600   = -599.0     分数变成巨大负值

乘数在 pop=5 处就已经穿过 0 并变负，排序被整体翻转：所有热门电影被推到列表
最末，只有交互次数接近 0 的冷门片能留在前排。这正是原推荐结果中
60 400 条推荐只覆盖 31 部电影、且全部是 Song of Freedom (1936) 这类
无人问津作品的原因，也解释了 Coverage=0.0084 与 Novelty=1.4427 两个异常值。

本实现改为：先把融合分归一化到 [0,1]，再做**加性**惩罚

    pop_norm = log1p(count) / log1p(max_count)   ∈ [0, 1]
    score    = score_norm - alpha * pop_norm

取对数是因为流行度服从长尾分布；改成加性是因为分数可能为负，
乘性惩罚对负分会产生"越惩罚越高"的反向效果。alpha 默认 0.10，
且有 alpha=0 的消融开关，惩罚强度对指标的影响可被直接测量。

**2. 归一化范围**

原实现在候选集上做 min-max，会被单个离群分数整体压扁。默认改为 z-score
（对离群更稳健），并保留 minmax / rank 两种可选，便于消融对比。
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Callable, Dict, List

import numpy as np

from .config import FusionConfig
from .data import Dataset

logger = logging.getLogger(__name__)

NEG_INF = np.float32(-np.inf)


# ---------------------------------------------------------------- 归一化
def normalize_scores(S: np.ndarray, valid_mask: np.ndarray, method: str) -> np.ndarray:
    """按用户（行）对候选物品的分数做归一化。非候选位置输出 0，不参与统计。

    S           : (n_users, n_items) 原始分数
    valid_mask  : (n_users, n_items) 布尔，True 表示该物品是该用户的候选
    """
    out = np.zeros_like(S, dtype=np.float32)
    X = np.where(valid_mask, S, np.nan).astype(np.float32)

    if method == "zscore":
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd[sd < 1e-8] = 1.0
        Z = (X - mu) / sd
    elif method == "minmax":
        lo = np.nanmin(X, axis=1, keepdims=True)
        hi = np.nanmax(X, axis=1, keepdims=True)
        rng = hi - lo
        rng[rng < 1e-8] = 1.0
        Z = (X - lo) / rng
    elif method == "rank":
        # 逐行分位数归一化：对分数尺度完全不敏感，代价是丢失强度信息
        Z = np.full_like(X, np.nan)
        for u in range(X.shape[0]):
            idx = np.flatnonzero(valid_mask[u])
            if idx.size == 0:
                continue
            order = np.argsort(np.argsort(X[u, idx]))
            Z[u, idx] = order / max(idx.size - 1, 1)
    else:
        raise ValueError(f"未知的归一化方式：{method}")

    np.copyto(out, np.nan_to_num(Z, nan=0.0), where=valid_mask)
    return out


# ---------------------------------------------------------------- 融合器
class HybridFusion:
    """把多路模型的打分矩阵融合为最终排序分。"""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.weights: Dict[str, float] = dict(cfg.weights)

    # ---------------------------------------------------------- 候选与掩码
    @staticmethod
    def candidate_mask(ds: Dataset, target: str = "test") -> np.ndarray:
        """按时间顺序构建离线排序候选集。

        Validation:
            只排除 train 历史。
            未来 test interaction 仍然未知，因此不得利用它构建候选集。

        Test:
            排除 train + validation 历史。
            validation interaction 在 test 时间点已经发生，因此属于已知历史。

        当前 held-out target 始终保留在候选集中。
        """
        if target not in {"valid", "test"}:
            raise ValueError("target must be 'valid' or 'test'")

        mask = np.ones((ds.n_users, ds.n_items), dtype=bool)

        mask[
            ds.train["u"].to_numpy(np.int64),
            ds.train["i"].to_numpy(np.int64),
        ] = False

        if target == "test":
            mask[
                ds.valid["u"].to_numpy(np.int64),
                ds.valid["i"].to_numpy(np.int64),
            ] = False

        target_df = ds.valid if target == "valid" else ds.test
        mask[
            target_df["u"].to_numpy(np.int64),
            target_df["i"].to_numpy(np.int64),
        ] = True

        return mask

    @staticmethod
    def popularity_norm(ds: Dataset) -> np.ndarray:
        pop = ds.item_popularity
        return (np.log1p(pop) / np.log1p(max(pop.max(), 1.0))).astype(np.float32)

    # ---------------------------------------------------------- 融合
    def fuse(self, scores: Dict[str, np.ndarray], ds: Dataset,
             valid_mask: np.ndarray | None = None,
             weights: Dict[str, float] | None = None,
             alpha: float | None = None) -> np.ndarray:
        """返回 (n_users, n_items) 的最终分数，非候选位置为 -inf。"""
        w = weights if weights is not None else self.weights
        a = self.cfg.popularity_alpha if alpha is None else alpha
        if valid_mask is None:
            valid_mask = self.candidate_mask(ds)

        total = np.zeros((ds.n_users, ds.n_items), dtype=np.float32)
        wsum = sum(w.get(k, 0.0) for k in scores)
        if wsum <= 0:
            raise ValueError("融合权重全为 0")
        for name, S in scores.items():
            wi = w.get(name, 0.0)
            if wi == 0:
                continue
            total += np.float32(wi / wsum) * normalize_scores(S, valid_mask, self.cfg.normalize)

        if a > 0:
            # 先把融合分压到 [0,1]，热度惩罚才有可比的量纲
            lo = np.where(valid_mask, total, np.inf).min(axis=1, keepdims=True)
            hi = np.where(valid_mask, total, -np.inf).max(axis=1, keepdims=True)
            rng = np.maximum(hi - lo, 1e-8)
            total = (total - lo) / rng
            total -= np.float32(a) * self.popularity_norm(ds)[None, :]

        total[~valid_mask] = NEG_INF
        return total

    # ---------------------------------------------------------- 权重搜索
    def search_weights(self, scores: Dict[str, np.ndarray], ds: Dataset,
                       objective: Callable[[np.ndarray, np.ndarray], float],
                       rng: np.random.Generator) -> Dict[str, float]:
        """在**验证集**上网格搜索融合权重。

        搜索只使用验证集；测试集在整个模型选择过程中从不参与。

        默认在全部验证用户上评估（``search_users = 0``）。早期版本只用 800 位
        采样用户，导致选择过程本身过拟合——受控实验（固定同一批打分矩阵、
        只重采验证用户子集）显示，模型参数一字未变，选出的 NeuMF 权重却在
        0.2 ~ 1.0 之间摆动。详见 ``config.FusionConfig`` 的说明与
        ``experiments/weight_search_stability.py``。

        objective(fused_scores, target_items) -> 越大越好。
        当 ``search_objective == "sampled"`` 时，改用 1 正 + 99 负的低方差
        代理目标，此时忽略传入的 objective。
        """
        names = [k for k in scores if self.cfg.weights.get(k, 0.0) is not None]
        step = self.cfg.search_step
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)

        v_u = ds.valid["u"].to_numpy()
        v_i = ds.valid["i"].to_numpy()
        valid_mask = self.candidate_mask(ds, target="valid")

        n_all = len(v_u)
        k = n_all if self.cfg.search_users <= 0 else min(self.cfg.search_users, n_all)
        if k < n_all:
            sel = rng.choice(n_all, size=k, replace=False)
            su, si = v_u[sel], v_i[sel]
            sub_mask = valid_mask[su]
            sub = {n: scores[n][su] for n in names}
        else:
            su, si = v_u, v_i
            sub_mask = valid_mask[su]
            sub = {n: scores[n][su] for n in names}

        # 采样负例代理目标：为每位验证用户预先抽好固定的 99 个负例，
        # 保证所有权重组合面对同一批负例，比较才是公平的。
        neg_items = None
        if self.cfg.search_objective == "sampled":
            n_neg = 99
            neg_items = np.empty((k, n_neg), dtype=np.int64)
            for r in range(k):
                pool = np.flatnonzero(sub_mask[r])
                pool = pool[pool != si[r]]

                if pool.size == 0:
                    raise ValueError(
                        f"Validation user row {r} has no valid negative candidates."
                    )

                neg_items[r] = rng.choice(
                    pool,
                    size=n_neg,
                    replace=pool.size < n_neg,
                )
        elif self.cfg.search_objective != "full":
            raise ValueError(f"未知的 search_objective：{self.cfg.search_objective}")

        combos: List[tuple] = [
            c for c in itertools.product(grid, repeat=len(names))
            if abs(sum(c) - 1.0) < 1e-6
        ]
        logger.info("[fusion] 权重网格搜索：%d 种组合 × %d 位验证用户（目标=%s）",
                    len(combos), k, self.cfg.search_objective)

        best_w, best_v = dict(self.weights), -np.inf
        pop = self.popularity_norm(ds)
        norm_cache = {n: normalize_scores(sub[n], sub_mask, self.cfg.normalize)
                      for n in names}
        t0 = time.time()

        for combo in combos:
            w = dict(zip(names, combo))
            total = np.zeros_like(norm_cache[names[0]])
            for n, wi in w.items():
                if wi:
                    total += np.float32(wi) * norm_cache[n]
            if self.cfg.popularity_alpha > 0:
                lo = np.where(sub_mask, total, np.inf).min(axis=1, keepdims=True)
                hi = np.where(sub_mask, total, -np.inf).max(axis=1, keepdims=True)
                total = (total - lo) / np.maximum(hi - lo, 1e-8)
                total = total - np.float32(self.cfg.popularity_alpha) * pop[None, :]
            total = np.where(sub_mask, total, -np.inf)

            if neg_items is None:
                v = objective(total, si)
            else:
                v = _sampled_ndcg(total, si, neg_items)

            if v > best_v:
                best_v, best_w = v, w

        logger.info("[fusion] 最优权重 %s（验证目标值=%.4f，耗时 %.0fs）",
                    {k2: round(v2, 2) for k2, v2 in best_w.items()},
                    best_v, time.time() - t0)
        self.weights = best_w
        self.search_objective_value_ = best_v
        return best_w


def _sampled_ndcg(scores: np.ndarray, pos: np.ndarray, negs: np.ndarray,
                  top_n: int = 10) -> float:
    """1 正 + N 负协议下的 nDCG@10，作为权重搜索的低方差代理目标。

    并列规则与 evaluate.positive_rank 保持一致：
    分数降序，完全同分时物品内部索引升序。
    """
    cand = np.concatenate([pos[:, None], negs], axis=1)
    s = np.take_along_axis(scores, cand, axis=1)
    worse = ((s[:, 1:] > s[:, :1])
             | ((s[:, 1:] == s[:, :1]) & (cand[:, 1:] < pos[:, None])))
    rank = worse.sum(axis=1)
    return float(np.where(rank < top_n, 1.0 / np.log2(rank + 2.0), 0.0).mean())
