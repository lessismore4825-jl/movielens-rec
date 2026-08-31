"""召回/评分模型的统一接口。

设计要点
--------
原实现的 `_predict_cf` / `_predict_svd` / `_predict_content` 都是"每个用户、每个候选
物品调用一次 Python 函数"，6040 用户 × 约 3700 候选 ≈ 2 200 万次调用，其中
`_predict_content` 还嵌套了一层用户历史循环，总计约 37 亿次操作——这才是 203 分钟
运行时间的真正来源，与有无 GPU 无关。

这里改为矩阵接口：每个模型一次性产出 (n_users, n_items) 的稠密打分矩阵。
ml-1m 规模下单个矩阵为 6040 × 3883 × 4B ≈ 94 MB，四路模型合计 <400 MB，
现代笔记本内存完全放得下，而计算量下降 2~3 个数量级。
"""
from __future__ import annotations

import abc
import logging
import time

import numpy as np

from ..data import Dataset

logger = logging.getLogger(__name__)


class BaseRecommender(abc.ABC):
    """所有召回模型的基类。"""

    name: str = "base"
    #: 该模型的输出是否为「可与真实评分直接比较的评分预测」。
    #: True  -> 参与 MAE / RMSE 等回归指标评估
    #: False -> 只输出排序偏好分（如 NeuMF 的隐式概率、内容相似度）
    predicts_ratings: bool = False

    @abc.abstractmethod
    def fit(self, ds: Dataset, valid_cb=None) -> "BaseRecommender":
        ...

    @abc.abstractmethod
    def predict_all(self, ds: Dataset) -> np.ndarray:
        """返回 (n_users, n_items) 的 float32 打分矩阵。"""

    def predict_pairs(self, ds: Dataset, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        """对给定 (u, i) 对打分。默认实现走全量矩阵，子类可覆写为更省内存的版本。"""
        return self.predict_all(ds)[u, i]

    # ------------------------------------------------------------ 计时辅助
    def timed_fit(self, ds: Dataset, **kw) -> "BaseRecommender":
        t0 = time.time()
        logger.info("[%s] 开始训练", self.name)
        self.fit(ds, **kw)
        self.fit_seconds = time.time() - t0
        logger.info("[%s] 训练完成，耗时 %.1fs", self.name, self.fit_seconds)
        return self


class PopularityRecommender(BaseRecommender):
    """最简单的非个性化基线：按训练集交互次数排序。

    加入它是为了给混合系统一个诚实的参照——如果混合系统打不过热门榜，
    那么"个性化"就没有发生。原项目缺少这一基线，导致无法察觉推荐结果
    其实已经退化为"所有用户几乎推荐同一批电影"。
    """

    name = "popularity"

    def fit(self, ds: Dataset, valid_cb=None) -> "PopularityRecommender":
        self._scores = np.log1p(ds.item_popularity).astype(np.float32)
        return self

    def predict_all(self, ds: Dataset) -> np.ndarray:
        return np.tile(self._scores, (ds.n_users, 1))


class RandomRecommender(BaseRecommender):
    """随机基线，用于确认评估协议本身没有把指标钉死在 0。"""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, ds: Dataset, valid_cb=None) -> "RandomRecommender":
        return self

    def predict_all(self, ds: Dataset) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random((ds.n_users, ds.n_items), dtype=np.float32)
