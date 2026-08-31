"""基于内容的推荐：TF-IDF（可选叠加 SBERT）+ 用户画像向量。

原实现的性能与建模问题
----------------------
原 `_predict_content` 的结构是：

    for mid in candidate_movies:              # 约 3 700 个候选
        for rated_mid in user_rated:          # 约 165 条历史
            sim = (tfidf_sim[idx, rated_idx] + sbert_sim[idx, rated_idx]) / 2

即每位用户约 61 万次内层循环，6040 位用户合计约 37 亿次 Python 级操作。
但这段代码在数学上等价于「候选物品向量 与 用户历史物品向量均值 的内积」，
可以整体写成一次矩阵乘法：

    user_profile = normalize(W_u @ X)         # (n_users, n_features)
    scores       = user_profile @ X.T         # (n_users, n_items)

复杂度从 O(U·C·H) 降到一次稀疏矩阵乘法，且完全不需要物化 3883×3883 的
相似度矩阵（原实现为 TF-IDF 和 SBERT 各存了一份，合计约 240 MB）。

建模上还做了一处改动：原实现对历史物品做**无权重**平均，等于认为用户打 1 分
和打 5 分的电影同样能代表他的口味。这里改为以「评分 − 用户均分」为权重，
让低于个人均分的电影产生负向贡献，画像才真正反映偏好而非曝光。
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..config import ContentConfig
from ..data import Dataset
from .base import BaseRecommender

logger = logging.getLogger(__name__)


class ContentRecommender(BaseRecommender):
    name = "content"

    def __init__(self, cfg: ContentConfig):
        self.cfg = cfg

    # ------------------------------------------------------------ 物品表征
    def _build_item_matrix(self, ds: Dataset):
        texts = ds.movies["combined_features"].fillna("").tolist()
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=self.cfg.tfidf_max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w'\-]+\b",
        )
        X = normalize(vec.fit_transform(texts)).astype(np.float32)
        logger.info("[content] TF-IDF 维度 %d × %d（非零 %.3f%%）",
                    X.shape[0], X.shape[1], X.nnz / (X.shape[0] * X.shape[1]) * 100)
        self.vectorizer_ = vec

        if not self.cfg.use_sbert:
            return X

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("[content] 未安装 sentence-transformers，跳过 SBERT，仅用 TF-IDF")
            return X
        try:
            model = SentenceTransformer(self.cfg.sbert_model)
            emb = normalize(model.encode(texts, show_progress_bar=False,
                                         batch_size=128).astype(np.float32))
        except Exception as exc:                       # 断网 / 模型下载失败
            logger.warning("[content] SBERT 加载失败（%s），降级为仅用 TF-IDF", exc)
            return X

        # 语义向量是稠密的，与稀疏 TF-IDF 按权重横向拼接；
        # 拼接后再整体归一化，等价于两种相似度的加权和。
        w = self.cfg.sbert_weight
        logger.info("[content] 叠加 SBERT 语义向量（权重 %.2f），维度 %d", w, emb.shape[1])
        return normalize(sparse.hstack([X * np.sqrt(1 - w),
                                        sparse.csr_matrix(emb * np.sqrt(w))]).tocsr())

    # ------------------------------------------------------------ 训练
    def fit(self, ds: Dataset, valid_cb=None) -> "ContentRecommender":
        self.X_ = self._build_item_matrix(ds)

        R = ds.train_csr                                   # (n_users, n_items)
        mask = ds.train_mask
        counts = np.asarray(mask.sum(axis=1)).ravel()
        sums = np.asarray(R.sum(axis=1)).ravel()
        user_mean = np.where(counts > 0, sums / np.maximum(counts, 1),
                             ds.train["rating"].mean())
        self.user_mean_ = user_mean.astype(np.float32)

        # 以「评分 − 用户均分」为权重构建画像，负偏好也参与
        W = R.copy().astype(np.float32)
        W.data -= np.repeat(self.user_mean_, np.diff(W.indptr))
        profile = W @ self.X_                              # (n_users, n_features)，保持稀疏
        self.profile_ = normalize(profile).astype(np.float32)
        return self

    def predict_all(self, ds: Dataset) -> np.ndarray:
        scores = self.profile_ @ self.X_.T                 # 余弦相似度（两侧均已 L2 归一）
        if sparse.issparse(scores):
            scores = scores.toarray()
        return np.asarray(scores, dtype=np.float32)
