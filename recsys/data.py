"""数据加载、索引重映射与留一法（leave-one-out）切分。

与原实现的关键差异
-------------------
1. 切分先于训练：train / valid / test 三份严格互斥，训练集绝不包含测试正例。
   原实现用全量 ratings 训练后又在其子集上评估，属于数据泄漏。
2. 按 (userId, timestamp) 排序后再取 tail，才是真正"用户最后一次行为"的留一法。
   原实现直接 groupby().tail(1)，取到的是文件行序的最后一条。
3. user / item 索引由同一个映射统一产生，不再依赖 users 与 ratings 各自
   astype('category').cat.codes 恰好对齐的巧合。
4. movies 中未被任何人评分的电影同样保留在物品空间内（原实现 dropna 直接丢弃），
   因为基于内容的召回本就要服务这部分冷启动物品。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .config import PathConfig

logger = logging.getLogger(__name__)

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class Dataset:
    """切分完成的数据集。所有 DataFrame 中的 u / i 均为 0 起始的连续索引。"""

    train: pd.DataFrame          # 列: u, i, rating, timestamp
    valid: pd.DataFrame          # 每用户 1 条（倒数第二次交互），用于早停与权重搜索
    test: pd.DataFrame           # 每用户 1 条（最后一次交互），用于最终评估
    movies: pd.DataFrame         # 索引为 i，含 title / genres / year / combined_features
    users: pd.DataFrame          # 索引为 u，含 gender / age / occupation
    n_users: int
    n_items: int
    user_id_map: Dict[int, int]  # 原始 userId -> u
    item_id_map: Dict[int, int]  # 原始 movieId -> i

    # ---- 派生结构（惰性构建） ----
    _train_csr: sparse.csr_matrix | None = None
    _train_mask: sparse.csr_matrix | None = None
    _item_pop: np.ndarray | None = None

    @property
    def train_csr(self) -> sparse.csr_matrix:
        """训练集评分矩阵 (n_users, n_items)，缺失值为显式 0。"""
        if self._train_csr is None:
            self._train_csr = sparse.csr_matrix(
                (self.train["rating"].to_numpy(np.float32),
                 (self.train["u"].to_numpy(), self.train["i"].to_numpy())),
                shape=(self.n_users, self.n_items),
            )
        return self._train_csr

    @property
    def train_mask(self) -> sparse.csr_matrix:
        """0/1 交互矩阵，用于区分"评了 0 分"与"没评过"。"""
        if self._train_mask is None:
            m = self.train_csr.copy()
            m.data = np.ones_like(m.data)
            self._train_mask = m
        return self._train_mask

    @property
    def item_popularity(self) -> np.ndarray:
        """每个物品在训练集中的交互次数，长度 n_items。"""
        if self._item_pop is None:
            pop = np.zeros(self.n_items, dtype=np.float64)
            counts = self.train["i"].value_counts()
            pop[counts.index.to_numpy()] = counts.to_numpy()
            self._item_pop = pop
        return self._item_pop

    def user_train_items(self) -> list[np.ndarray]:
        """每个用户在训练集中交互过的物品索引（升序），用于候选集生成。"""
        indptr, indices = self.train_mask.indptr, self.train_mask.indices
        return [indices[indptr[u]:indptr[u + 1]] for u in range(self.n_users)]

    def describe(self) -> str:
        density = len(self.train) / (self.n_users * self.n_items) * 100
        return (
            f"用户 {self.n_users} / 物品 {self.n_items} / "
            f"训练 {len(self.train):,} 验证 {len(self.valid):,} 测试 {len(self.test):,} "
            f"(训练集稠密度 {density:.3f}%)"
        )


class DataLoader:
    """从 ml-1m 的三个 .dat 文件构建 Dataset。"""

    def __init__(self, paths: PathConfig):
        self.paths = paths

    # ------------------------------------------------------------ 原始读取
    def _read_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.paths.validate()
        ratings = pd.read_csv(
            self.paths.ratings, sep="::", engine="python", encoding="latin-1",
            names=["userId", "movieId", "rating", "timestamp"],
            dtype={"userId": "int32", "movieId": "int32",
                   "rating": "float32", "timestamp": "int64"},
        )
        movies = pd.read_csv(
            self.paths.movies, sep="::", engine="python", encoding="latin-1",
            names=["movieId", "title", "genres"],
            dtype={"movieId": "int32", "title": "str", "genres": "str"},
        )
        users = pd.read_csv(
            self.paths.users, sep="::", engine="python", encoding="latin-1",
            names=["userId", "gender", "age", "occupation", "zipcode"],
            dtype={"userId": "int32", "gender": "str", "age": "int16",
                   "occupation": "int16", "zipcode": "str"},
        )
        return ratings, movies, users

    # ------------------------------------------------------------ 特征工程
    @staticmethod
    def _build_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
        movies = movies.copy()
        movies["year"] = (
            movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("Float64")
        )
        movies["clean_title"] = (
            movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
        )
        movies["genres"] = movies["genres"].fillna("Unknown")
        # 十年档作为一个粗粒度的时代特征，让内容相似度不只看类型词
        decade = (movies["year"] // 10 * 10)
        movies["decade"] = decade.astype("Int64").astype("string").fillna("unknown")
        movies["year"] = movies["year"].fillna(-1).astype("int32")
        movies["combined_features"] = (
            movies["genres"].str.replace("|", " ", regex=False)
            + " " + movies["clean_title"]
            + " decade_" + movies["decade"]
        )
        return movies

    @staticmethod
    def _genre_matrix(movies: pd.DataFrame) -> np.ndarray:
        """(n_items, n_genres) 的 0/1 矩阵，供多样性指标使用。"""
        g = np.zeros((len(movies), len(GENRE_LIST)), dtype=np.float32)
        idx = {name: k for k, name in enumerate(GENRE_LIST)}
        for row, raw in enumerate(movies["genres"].to_numpy()):
            for name in str(raw).split("|"):
                k = idx.get(name)
                if k is not None:
                    g[row, k] = 1.0
        return g

    # ------------------------------------------------------------ 主流程
    def load(self, min_interactions: int = 5) -> Dataset:
        ratings, movies, users = self._read_raw()
        logger.info("原始数据：评分 %s 条 / 电影 %s 部 / 用户 %s 人",
                    f"{len(ratings):,}", f"{len(movies):,}", f"{len(users):,}")

        # 过滤交互过少的用户（ml-1m 本身保证 >=20，此处为通用性保留）
        counts = ratings["userId"].value_counts()
        keep = counts[counts >= min_interactions].index
        dropped = len(counts) - len(keep)
        if dropped:
            logger.info("过滤掉交互数 < %d 的用户 %d 人", min_interactions, dropped)
            ratings = ratings[ratings["userId"].isin(keep)]

        movies = self._build_movie_features(movies)

        # ---- 统一索引映射：物品空间 = movies.dat 的全部电影 ----
        item_ids = np.sort(movies["movieId"].unique())
        item_id_map = {int(m): k for k, m in enumerate(item_ids)}
        user_ids = np.sort(ratings["userId"].unique())
        user_id_map = {int(u): k for k, u in enumerate(user_ids)}

        # ratings 中可能出现 movies.dat 里没有的 movieId，直接丢弃（ml-1m 中为 0 条）
        unknown = ~ratings["movieId"].isin(item_id_map)
        if unknown.any():
            logger.warning("丢弃 %d 条指向未知 movieId 的评分", int(unknown.sum()))
            ratings = ratings[~unknown]

        ratings = ratings.assign(
            u=ratings["userId"].map(user_id_map).astype("int32"),
            i=ratings["movieId"].map(item_id_map).astype("int32"),
        )

        movies = movies.assign(i=movies["movieId"].map(item_id_map).astype("int32"))
        movies = movies.set_index("i").sort_index()
        movies.attrs["genre_matrix"] = self._genre_matrix(movies)

        users = users[users["userId"].isin(user_id_map)].copy()
        users["u"] = users["userId"].map(user_id_map).astype("int32")
        users = users.set_index("u").sort_index()

        train, valid, test = self._leave_one_out(ratings)

        ds = Dataset(
            train=train, valid=valid, test=test, movies=movies, users=users,
            n_users=len(user_id_map), n_items=len(item_id_map),
            user_id_map=user_id_map, item_id_map=item_id_map,
        )
        logger.info("切分完成：%s", ds.describe())
        self._assert_no_leakage(ds)
        return ds

    # ------------------------------------------------------------ 留一法
    @staticmethod
    def _leave_one_out(ratings: pd.DataFrame):
        """按时间排序后，每位用户最后一条进 test、倒数第二条进 valid、其余进 train。"""
        cols = ["u", "i", "rating", "timestamp"]
        # 以 timestamp 为主键、原始行序为次键排序，保证同一秒内的多条交互顺序稳定
        r = ratings[cols].copy()
        r["_row"] = np.arange(len(r), dtype=np.int64)
        r = r.sort_values(["u", "timestamp", "_row"], kind="mergesort")

        rank_from_end = r.groupby("u", sort=False).cumcount(ascending=False)
        test = r[rank_from_end == 0]
        valid = r[rank_from_end == 1]
        train = r[rank_from_end >= 2]
        return (train[cols].reset_index(drop=True),
                valid[cols].reset_index(drop=True),
                test[cols].reset_index(drop=True))

    @staticmethod
    def _assert_no_leakage(ds: Dataset) -> None:
        """显式断言训练集与验证/测试集无交集——原实现正是栽在这里。"""
        train_pairs = set(map(tuple, ds.train[["u", "i"]].to_numpy()))
        for name, split in (("验证集", ds.valid), ("测试集", ds.test)):
            overlap = sum(1 for p in map(tuple, split[["u", "i"]].to_numpy())
                          if p in train_pairs)
            if overlap:
                raise AssertionError(f"{name}有 {overlap} 条 (u,i) 同时出现在训练集中，存在数据泄漏")
        logger.info("泄漏自检通过：train ∩ valid = train ∩ test = ∅")
