"""评估层：留一法排序指标、全候选排序指标与评分回归指标。

原实现的评估缺陷清单
--------------------
1. **正例被排除在候选集之外**（致命）。候选集用全量 ratings 生成，测试正例作为
   "已看过"被剔除，`full_scores.get(pos_item, 0)` 于是永远取到默认值 0，
   Precision@N / Recall@N / F1@N / nDCG@N / HR@N 五项结构性恒为 0。
2. **AUC 因此被虚高**。正例拿到默认分 0，而负例受热度惩罚 bug 影响普遍为负数，
   "0 分高于负分"把 AUC 抬到 0.9127。报告中"AUC 高而 HR=0 的两极分化现象"
   实际上是同一个 bug 的两种表现，而非真实的模型特性。
3. **IDCG 算错**。原代码 `ideal_rel` 从推荐列表 `ranked_ids` 里取，
   而理想排序应当来自真实相关物品集合。
4. **Recall 与 Precision 定义不一致**。Precision 用命中个数，Recall 却用二值 hit。
5. **Novelty 会溢出**。`1/np.log(1+p)` 在 p=0 时为 inf、p=1 时分母 log(2) 极小。
   本实现改用标准自信息 −log2(pop/|U|)。
6. **死代码**。`precision_recall_curve` 的返回值算完从未使用。

此外新增两项诊断指标：Gini 系数与 Top-1% 物品占比，用来量化"推荐结果是否
坍缩到少数几部电影"——原项目 60 400 条推荐只覆盖 31 部电影，正需要这类指标
在评估阶段就报警，而不是等人工翻 Excel 才发现。
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

from .data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 基础工具
def top_n_items(scores: np.ndarray, top_n: int) -> np.ndarray:
    """逐行返回 Top-N 物品。

    排序规则固定为：
      1. score descending
      2. internal item index ascending for exact ties

    使用 stable sort 保证并列分数的处理与 positive_rank 完全一致，
    避免 argpartition 在 tie block 中产生未定义/不一致的选择。
    """
    n = min(top_n, scores.shape[1])
    return np.argsort(-scores, axis=1, kind="stable")[:, :n]


def positive_rank(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """正例在候选集中的 0 起始确定性排名。

    Primary key:
        score descending

    Secondary key for exact ties:
        internal item index ascending

    因此与正例同分但 item index 更小的候选会排在正例之前。
    不能再把所有并列正例自动视为 rank 1。
    """
    rows = np.arange(len(pos))
    pos = np.asarray(pos, dtype=np.int64)
    pos_score = scores[rows, pos][:, None]

    item_idx = np.arange(scores.shape[1], dtype=np.int64)[None, :]

    strictly_better = scores > pos_score
    tied_before = (scores == pos_score) & (item_idx < pos[:, None])

    return (strictly_better | tied_before).sum(axis=1)


def gini(counts: np.ndarray) -> float:
    """推荐频次分布的 Gini 系数：0=完全均匀，1=全部集中在一个物品。"""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


# ---------------------------------------------------------------- 多样性/新颖性
def intra_list_diversity(rec: np.ndarray, genre_matrix: np.ndarray) -> float:
    """列表内多样性 = 推荐列表中两两电影的平均类型 Jaccard 距离。"""
    G = genre_matrix
    vals = []
    for row in rec:
        g = G[row]                                   # (top_n, n_genres) 0/1
        inter = g @ g.T
        sizes = g.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        iu = np.triu_indices(len(row), k=1)
        if iu[0].size:
            vals.append(float(1.0 - jac[iu].mean()))
    return float(np.mean(vals)) if vals else 0.0


def novelty(rec: np.ndarray, popularity: np.ndarray, n_users: int) -> float:
    """标准自信息新颖性：−log2(p(i))，p(i)=交互该物品的用户占比。值越大越冷门。"""
    p = np.clip(popularity / max(n_users, 1), 1.0 / max(n_users, 1), 1.0)
    return float((-np.log2(p))[rec].mean())


# ---------------------------------------------------------------- 排序指标
def full_ranking_metrics(scores: np.ndarray, pos: np.ndarray, ds: Dataset,
                         top_n: int = 10) -> Dict[str, float]:
    """全候选排序协议：在该用户所有未交互物品中排序，最严格也最贴近线上场景。

    scores 中非候选位置须已置为 -inf；正例必须在候选集内（由 fusion.candidate_mask 保证）。
    """
    n_eval = scores.shape[0]
    rank = positive_rank(scores, pos)
    hit = (rank < top_n).astype(np.float64)

    # 留一法下每位用户只有 1 个相关物品，故 Recall@N == HR@N，Precision@N == hit/N。
    precision = hit / top_n
    recall = hit.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    # 单相关物品的 IDCG = 1/log2(2) = 1，故 nDCG = DCG
    ndcg = np.where(rank < top_n, 1.0 / np.log2(rank + 2.0), 0.0)
    mrr = 1.0 / (rank + 1.0)

    rec = top_n_items(scores, top_n)
    flat = rec.ravel()
    counts = np.bincount(flat, minlength=ds.n_items)
    n_covered = int((counts > 0).sum())
    top1pct = max(1, ds.n_items // 100)
    share = counts[np.argsort(-counts)[:top1pct]].sum() / max(counts.sum(), 1)

    return {
        f"Precision@{top_n}": float(precision.mean()),
        f"Recall@{top_n}": float(recall.mean()),
        f"F1@{top_n}": float(f1.mean()),
        f"nDCG@{top_n}": float(ndcg.mean()),
        f"HR@{top_n}": float(hit.mean()),
        "MRR": float(mrr.mean()),
        "MedianRank": float(np.median(rank) + 1),
        "Coverage": n_covered / ds.n_items,
        "CoveredItems": float(n_covered),
        "Diversity": intra_list_diversity(rec, ds.movies.attrs["genre_matrix"]),
        "Novelty": novelty(rec, ds.item_popularity, ds.n_users),
        "Gini": gini(counts),
        "Top1%ItemShare": float(share),
        "EvalUsers": float(n_eval),
    }


def sampled_ranking_metrics(scores: np.ndarray, pos: np.ndarray, ds: Dataset,
                            rng: np.random.Generator, top_n: int = 10,
                            n_neg: int = 99) -> Dict[str, float]:
    """采样负例协议（He et al. 2017）：1 个正例 + n_neg 个未交互负例。

    这是 NeuMF / BPR 等论文报告 HR@10、nDCG@10 时的通用口径，
    数值上比全候选排序宽松得多，列出它是为了能与公开文献横向对比。
    AUC 在这里才是有意义的——它衡量正例排在随机负例之前的概率。
    """
    n = scores.shape[0]
    cand = np.empty((n, n_neg + 1), dtype=np.int64)
    cand[:, 0] = pos
    valid = np.isfinite(scores)
    for r in range(n):
        pool = np.flatnonzero(valid[r])
        pool = pool[pool != pos[r]]
        take = min(n_neg, len(pool))
        cand[r, 1:take + 1] = rng.choice(pool, size=take, replace=False)
        if take < n_neg:
            cand[r, take + 1:] = pos[r]      # 极端情况兜底，几乎不会触发

    s = np.take_along_axis(scores, cand, axis=1)

    # 与 full-candidate 协议使用同一个确定性 tie policy：
    # score descending；exact tie 时 internal item index ascending。
    neg_scores = s[:, 1:]
    pos_scores = s[:, :1]
    neg_items = cand[:, 1:]

    rank = (
        (neg_scores > pos_scores)
        | ((neg_scores == pos_scores) & (neg_items < pos[:, None]))
    ).sum(axis=1)

    hit = (rank < top_n).astype(np.float64)
    ndcg = np.where(rank < top_n, 1.0 / np.log2(rank + 2.0), 0.0)
    # AUC = 正例分数严格高于负例的比例（并列记 0.5）
    diff = s[:, :1] - s[:, 1:]
    auc = ((diff > 0).sum(axis=1) + 0.5 * (diff == 0).sum(axis=1)) / diff.shape[1]
    return {
        f"HR@{top_n}(sampled)": float(hit.mean()),
        f"nDCG@{top_n}(sampled)": float(ndcg.mean()),
        "MRR(sampled)": float((1.0 / (rank + 1.0)).mean()),
        "AUC(sampled)": float(auc.mean()),
    }


# ---------------------------------------------------------------- 回归指标
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       rating_range: float = 4.0) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.abs(err).mean())
    mse = float((err ** 2).mean())
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse,
            "NMAE": mae / rating_range, "NRMSE": rmse / rating_range}


def evaluate_rating_models(models: Sequence, ds: Dataset,
                           preds: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """在**测试集**上评估评分预测能力。

    与原实现的区别：这里的测试集从未参与训练。原实现用
    `data.build_full_trainset()` 在全量评分上训练，再在其子集上测 MAE/RMSE，
    测试样本 100% 已被模型见过，报告中的 MAE=0.586 等数值因此偏乐观，不可用作结论。
    """
    u = ds.test["u"].to_numpy()
    i = ds.test["i"].to_numpy()
    y = ds.test["rating"].to_numpy(np.float64)

    out: Dict[str, Dict[str, float]] = {}
    for m in models:
        if not getattr(m, "predicts_ratings", False):
            continue
        out[m.name] = regression_metrics(y, preds[m.name][u, i].astype(np.float64))

    # 全局均分基线：任何评分预测模型都必须显著优于它，否则说明没学到东西
    out["global_mean(baseline)"] = regression_metrics(
        y, np.full_like(y, ds.train["rating"].mean()))
    return out


# ---------------------------------------------------------------- 汇总打印
def format_metrics(title: str, metrics: Dict[str, float]) -> str:
    lines = [f"── {title} " + "─" * max(0, 60 - len(title))]
    for k, v in metrics.items():
        lines.append(f"   {k:<24s} {v:>12.4f}")
    return "\n".join(lines)
