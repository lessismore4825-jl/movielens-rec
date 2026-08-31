"""基于物品的协同过滤（Item-based KNN）。

为什么从 user-based 换成 item-based
------------------------------------
原实现用 surprise 的 `KNNBasic(user_based=True)`。在 ml-1m 上 user-based 需要维护
6040×6040 的用户相似度，且每来一个候选物品都要重新扫描邻居；item-based 只需
3883×3883 的物品相似度，且物品相似度比用户相似度稳定得多（物品的评分分布不随
时间快速漂移）。更重要的是，item-based 的打分可以写成一次稀疏矩阵乘法，
对全体用户一次算完。

打分公式（去中心化的加权平均）
    r̂(u,i) = μ_i + Σ_{j∈N(i)∩R(u)} s(i,j)·(r_uj − μ_j) / (Σ_{j} |s(i,j)| + ε)

相似度采用共评人数收缩（shrinkage）的余弦：
    s(i,j) = cos(i,j) · n_ij / (n_ij + λ)
n_ij 为同时评过 i 和 j 的用户数。原实现的 `min_support=5` 是硬截断，
收缩项是它的连续版本，能避免"只有 5 个共同评分者却相似度 1.0"的虚高邻居。
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from ..config import ItemKNNConfig
from ..data import Dataset
from .base import BaseRecommender

logger = logging.getLogger(__name__)


class ItemKNNRecommender(BaseRecommender):
    name = "itemknn"
    predicts_ratings = True

    def __init__(self, cfg: ItemKNNConfig):
        self.cfg = cfg

    def fit(self, ds: Dataset, valid_cb=None) -> "ItemKNNRecommender":
        R = ds.train_csr.tocsc()          # (n_users, n_items)
        mask = ds.train_mask.tocsc()

        # 物品均值（只对评过的用户求均值，缺失值不计入分母）
        counts = np.asarray(mask.sum(axis=0)).ravel()
        sums = np.asarray(R.sum(axis=0)).ravel()
        global_mean = float(ds.train["rating"].mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            item_mean = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)
        self.item_mean_ = item_mean.astype(np.float32)
        self.global_mean_ = np.float32(global_mean)

        # 去中心化：只在已评分位置减去物品均值，未评分位置保持结构性 0
        Rc = R.copy().astype(np.float32)
        Rc.data -= np.repeat(self.item_mean_, np.diff(Rc.indptr))

        # 余弦相似度 + 共评人数收缩
        norms = np.sqrt(np.asarray(Rc.multiply(Rc).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        Rn = Rc.multiply(sparse.csr_matrix(1.0 / norms)).tocsc()
        S = np.asarray((Rn.T @ Rn).todense(), dtype=np.float32)     # (n_items, n_items)

        co_counts = np.asarray((mask.T @ mask).todense(), dtype=np.float32)
        S *= co_counts / (co_counts + np.float32(self.cfg.shrinkage))
        S[co_counts < self.cfg.min_support] = 0.0
        np.fill_diagonal(S, 0.0)

        # 每个物品只保留 top-k 个最相似的邻居，其余置零（降噪 + 稀疏化）
        k = min(self.cfg.topk, S.shape[1] - 1)
        if k > 0:
            cut = np.partition(np.abs(S), -k, axis=1)[:, -k][:, None]
            S[np.abs(S) < cut] = 0.0
        self.S_ = sparse.csr_matrix(S)
        logger.info("[itemknn] 相似度矩阵稀疏度 %.4f%%（top-%d 邻居）",
                    self.S_.nnz / S.size * 100, k)

        self._Rc = Rc.tocsr()
        self._mask = ds.train_mask
        return self

    def predict_all(self, ds: Dataset) -> np.ndarray:
        S_abs = abs(self.S_)
        numer = np.asarray((self._Rc @ self.S_.T).todense(), dtype=np.float32)
        denom = np.asarray((self._mask @ S_abs.T).todense(), dtype=np.float32)
        out = self.item_mean_[None, :] + numer / (denom + np.float32(1e-8))
        # 邻居完全缺失时回退到物品均值，避免出现 NaN
        out[denom < 1e-8] = np.broadcast_to(self.item_mean_, out.shape)[denom < 1e-8]
        return np.clip(out, 1.0, 5.0, out=out)


class ItemCFImplicitRecommender(BaseRecommender):
    """隐式反馈版 Item-CF：只看"看没看过"，不看打了几分。

    加这一路是因为原项目（以及很多课程实现）默认了一个并不成立的前提：
    **评分预测准 ⇒ 排序好**。实际上评分预测模型会把"只有 3 个人评过、
    但这 3 个人都打了 5 分"的冷门片预测成 4.9 分，于是 Top-N 列表被
    低置信度的长尾物品占满——本项目的实验里，RMSE 最好的 ItemKNN(0.957)
    全候选 HR@10 只有 0.019，反而不如"按热门排序"这条基线(0.037)。

    隐式 Item-CF 直接优化"共现"信号：

        s(i,j) = |U_i ∩ U_j| / (‖U_i‖·‖U_j‖) · n_ij/(n_ij+λ)
        score(u,i) = Σ_{j ∈ R(u)} s(i,j)

    它不产生评分，只产生排序偏好分，因此不参与 MAE/RMSE 评估。
    """

    name = "itemcf"
    predicts_ratings = False

    def __init__(self, cfg: ItemKNNConfig):
        self.cfg = cfg

    def fit(self, ds: Dataset, valid_cb=None) -> "ItemCFImplicitRecommender":
        B = ds.train_mask.astype(np.float32).tocsc()          # 0/1 交互矩阵
        counts = np.asarray(B.sum(axis=0)).ravel()
        norms = np.sqrt(np.maximum(counts, 1.0)).astype(np.float32)

        co = np.asarray((B.T @ B).todense(), dtype=np.float32)  # 共现次数 n_ij
        S = co / (norms[:, None] * norms[None, :])              # 余弦
        S *= co / (co + np.float32(self.cfg.shrinkage))         # 共现收缩
        S[co < self.cfg.min_support] = 0.0
        np.fill_diagonal(S, 0.0)

        k = min(self.cfg.topk, S.shape[1] - 1)
        if k > 0:
            cut = np.partition(S, -k, axis=1)[:, -k][:, None]
            S[S < cut] = 0.0
        self.S_ = sparse.csr_matrix(S)
        logger.info("[itemcf] 隐式相似度稀疏度 %.4f%%（top-%d 邻居）",
                    self.S_.nnz / S.size * 100, k)
        self._B = ds.train_mask.astype(np.float32)
        return self

    def predict_all(self, ds: Dataset) -> np.ndarray:
        return np.asarray((self._B @ self.S_.T).todense(), dtype=np.float32)
