"""NeuMF（GMF + MLP 双路融合），采用隐式反馈 + 负采样 + BCE 的标准训练目标。

原实现最关键的建模错误
----------------------
原代码把 NeuMF 当成评分回归模型来训练：

    ratings_norm = ratings['rating'].values / 5.0
    output = Dense(1, activation='sigmoid')(merged)
    model.compile(loss='mse')

它**只喂了观测到的 100 万条正样本**，没有任何负采样。这意味着模型从未见过
"用户不会看某部电影"的信号，最优解就是对任何 (u, i) 都输出接近全局均分的常数。
于是在 Top-N 排序时 NCF 这一路几乎是常数项，0.4 的权重实际贡献的是噪声——
这正好解释了报告里"NCF 的 MAE(0.724) 反而不如传统 CF(0.586)"这一反常现象。

He et al. (2017) 的原始设定是：把交互看作隐式反馈，正样本 label=1，
每个正样本再从未交互物品中采 k 个负样本 label=0，用 BCE 训练。本实现照此修正。

另外两处修正
------------
* 原实现两条路径的 embedding 维度都写成 128，且 MLP 首层 512，
  在 ml-1m（6040×3883）上参数量约 250 万，对 100 万条交互严重过参数化。
  这里按论文比例改为 GMF 32 维 + MLP 32 维、隐藏层 [64,32,16]。
* 原实现的 EarlyStopping 监控 `val_loss`，而 val 来自 `validation_split=0.1`
  的随机切分，同样落在训练分布内；这里改为监控验证集（每用户倒数第二次交互）
  的 HR@10，与最终评估目标一致。
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from ..config import NeuMFConfig
from ..data import Dataset
from ..device import resolve_device
from .base import BaseRecommender

logger = logging.getLogger(__name__)


class _NeuMFModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, cfg: NeuMFConfig):
        super().__init__()
        self.mf_u = nn.Embedding(n_users, cfg.mf_dim)
        self.mf_i = nn.Embedding(n_items, cfg.mf_dim)
        self.mlp_u = nn.Embedding(n_users, cfg.mlp_dim)
        self.mlp_i = nn.Embedding(n_items, cfg.mlp_dim)

        layers, in_dim = [], cfg.mlp_dim * 2
        for h in cfg.mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.mf_dim + in_dim, 1)

        for emb in (self.mf_u, self.mf_i, self.mlp_u, self.mlp_i):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        gmf = self.mf_u(u) * self.mf_i(i)                       # 逐元素积（GMF 路径）
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=-1))
        return self.head(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)   # 输出 logit


class NeuMFRecommender(BaseRecommender):
    name = "neumf"
    predicts_ratings = False    # 输出的是交互概率，不是评分，不参与 MAE/RMSE

    def __init__(self, cfg: NeuMFConfig, device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.seed = seed

    # ------------------------------------------------------------ 负采样
    @staticmethod
    def _known_positive_sets(
        ds: Dataset,
        *,
        include_valid: bool = False,
    ) -> list[set[int]]:
        """构建某个时间点已经可观察到的用户正交互集合。

        训练阶段只使用 train 历史。
        Validation 评估可以额外包含当前 validation interaction。

        Test interaction 永远不会用于训练或 validation-time sampling，
        避免未来信息进入模型选择过程。
        """
        positives = [set(items.tolist()) for items in ds.user_train_items()]

        if include_valid:
            for u, i in zip(
                ds.valid["u"].to_numpy(np.int64),
                ds.valid["i"].to_numpy(np.int64),
            ):
                positives[int(u)].add(int(i))

        return positives

    @staticmethod
    def _sample_negatives(
        pos_u: np.ndarray,
        n_items: int,
        k: int,
        blocked: list[set[int]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """为每个正样本采 k 个不属于该用户已知正交互集合的负样本。

        两段式策略，兼顾速度与正确性：

        1. **快路径**——均匀采样，然后只对撞上历史的位置重采，至多
           ``max_rounds`` 轮。训练集稠密度仅 4.2%，一轮之后残留冲突通常
           已低于 0.2%，因此绝大多数样本在这里就完成了。
        2. **兜底**——对仍然冲突的少数位置，逐用户从"未交互物品"的精确
           补集中采样。

        为什么必须有第 2 步：早期版本用无上界的 ``while True`` 一直重采到
        零冲突为止，遇到交互过全部物品的用户就会死循环；而只加上界、不做
        兜底又会把这些位置留成伪负例（标签错误的训练样本）。两段式让最坏
        情况的耗时有界，同时保证返回的每一个负例都是真负例。
        """
        max_rounds = 4
        m = len(pos_u) * k
        u_rep = np.repeat(pos_u, k)
        neg = rng.integers(0, n_items, size=m, dtype=np.int64)

        idx = np.arange(m)                       # 当前仍需检查的位置
        for _ in range(max_rounds):
            bad_local = np.fromiter(
                (int(i) in blocked[int(u)] for u, i in zip(u_rep[idx], neg[idx])),
                dtype=bool,
                count=len(idx),
            )
            idx = idx[bad_local]                 # 只保留仍然冲突的位置
            if idx.size == 0:
                return neg
            neg[idx] = rng.integers(0, n_items, size=idx.size, dtype=np.int64)

        # ---- 兜底：按用户分组，从精确补集中采样 ----
        logger.debug("[neumf] %d/%d (%.4f%%) 个负例进入精确补集兜底",
                     idx.size, m, idx.size / m * 100)
        all_items = np.arange(n_items, dtype=np.int64)
        order = np.argsort(u_rep[idx], kind="stable")
        idx_sorted = idx[order]
        users_sorted = u_rep[idx_sorted]
        bounds = np.flatnonzero(np.diff(users_sorted)) + 1

        for chunk in np.split(idx_sorted, bounds):
            if chunk.size == 0:
                continue
            u = int(u_rep[chunk[0]])
            pool = np.setdiff1d(all_items,
                                np.fromiter(blocked[u], dtype=np.int64,
                                            count=len(blocked[u])),
                                assume_unique=False)
            if pool.size == 0:
                raise ValueError(
                    f"User {u} has interacted with all {n_items} items; "
                    "no valid negative candidate exists."
                )
            neg[chunk] = rng.choice(pool, size=chunk.size,
                                    replace=chunk.size > pool.size)
        return neg

    def fit(self, ds: Dataset, valid_cb=None) -> "NeuMFRecommender":
        cfg, dev = self.cfg, self.device
        torch.manual_seed(self.seed)
        self.model = _NeuMFModule(ds.n_users, ds.n_items, cfg).to(dev)

        pos_u = ds.train["u"].to_numpy(np.int64)
        pos_i = ds.train["i"].to_numpy(np.int64)
        # 训练负采样只能使用训练时已经观察到的交互。
        blocked = self._known_positive_sets(ds)
        rng = np.random.default_rng(self.seed)

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad = -1.0, None, 0
        best_epoch, ep = 0, -1
        self.valid_hr_history_: list[float] = []

        # 验证集：每用户 1 个正例 + 99 个负例，监控 HR@10
        v_u, v_pos, v_neg = self._build_valid(ds, rng)

        for ep in range(cfg.epochs):
            neg_i = self._sample_negatives(
                pos_u,
                ds.n_items,
                cfg.n_negatives,
                blocked,
                rng,
            )
            u_all = np.concatenate([pos_u, np.repeat(pos_u, cfg.n_negatives)])
            i_all = np.concatenate([pos_i, neg_i])
            y_all = np.concatenate([np.ones(len(pos_i), np.float32),
                                    np.zeros(len(neg_i), np.float32)])
            perm = rng.permutation(len(u_all))
            u_all, i_all, y_all = u_all[perm], i_all[perm], y_all[perm]

            self.model.train()
            total, n = 0.0, len(u_all)
            for s in range(0, n, cfg.batch_size):
                e = s + cfg.batch_size
                ub = torch.as_tensor(u_all[s:e], device=dev)
                ib = torch.as_tensor(i_all[s:e], device=dev)
                yb = torch.as_tensor(y_all[s:e], device=dev)
                loss = lossf(self.model(ub, ib), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * (min(e, n) - s)

            hr = self._valid_hr(v_u, v_pos, v_neg)
            logger.info("[neumf] epoch %d/%d  bce=%.4f  valid_HR@10=%.4f",
                        ep + 1, cfg.epochs, total / n, hr)

            self.valid_hr_history_.append(hr)
            if hr > best + 1e-4:
                best, bad = hr, 0
                best_epoch = ep + 1
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    logger.info("[neumf] 验证 HR 连续 %d 轮无提升，早停于第 %d 轮", bad, ep + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_valid_hr_ = best
        self.best_epoch_ = best_epoch
        self.stopped_epoch_ = ep + 1
        # 早停点由验证 HR 的逐轮数值决定，而浮点累加顺序在不同平台/BLAS 上
        # 会有微小差异，因此这两个数字是跨平台复现比对的关键证据，
        # 需要一并写进 metrics.json（见 pipeline.train_and_score）。
        logger.info("[neumf] 最佳轮次 %d（valid HR@10=%.4f），实际训练至第 %d 轮",
                    best_epoch, best, ep + 1)
        return self

    # ------------------------------------------------------------ 验证辅助
    @staticmethod
    def _build_valid(ds: Dataset, rng: np.random.Generator, n_neg: int = 99):
        # Validation-time information = train history + validation target.
        # Test interaction is intentionally NOT used here.
        blocked = NeuMFRecommender._known_positive_sets(
            ds,
            include_valid=True,
        )

        u = ds.valid["u"].to_numpy(np.int64)
        pos = ds.valid["i"].to_numpy(np.int64)
        neg = np.empty((len(u), n_neg), dtype=np.int64)

        for row, uu in enumerate(u):
            banned = blocked[int(uu)]

            sampled: set[int] = set()

            while len(sampled) < n_neg:
                candidates = rng.integers(
                    0,
                    ds.n_items,
                    size=max(n_neg * 2, 64),
                    dtype=np.int64,
                )

                for i in candidates:
                    ii = int(i)

                    if ii not in banned:
                        sampled.add(ii)

                    if len(sampled) == n_neg:
                        break

            neg[row] = np.fromiter(
                sampled,
                dtype=np.int64,
                count=n_neg,
            )

        return u, pos, neg

    @torch.no_grad()
    def _valid_hr(self, u: np.ndarray, pos: np.ndarray, neg: np.ndarray, top_n: int = 10) -> float:
        self.model.eval()
        dev = self.device
        items = np.concatenate([pos[:, None], neg], axis=1)          # (N, 1+99)
        uu = torch.as_tensor(np.repeat(u, items.shape[1]), device=dev)
        ii = torch.as_tensor(items.ravel(), device=dev)
        scores = self.model(uu, ii).view(items.shape).cpu().numpy()
        # 与统一评估协议保持一致：
        # score descending；exact tie 时 internal item index ascending。
        neg_scores = scores[:, 1:]
        pos_scores = scores[:, :1]

        rank = (
            (neg_scores > pos_scores)
            | ((neg_scores == pos_scores) & (neg < pos[:, None]))
        ).sum(axis=1)

        return float((rank < top_n).mean())

    # ------------------------------------------------------------ 全量打分
    @torch.no_grad()
    def predict_all(self, ds: Dataset, user_batch: int = 64) -> np.ndarray:
        self.model.eval()
        dev = self.device
        out = np.empty((ds.n_users, ds.n_items), dtype=np.float32)
        all_i = torch.arange(ds.n_items, device=dev)
        for s in range(0, ds.n_users, user_batch):
            us = torch.arange(s, min(s + user_batch, ds.n_users), device=dev)
            b = len(us)
            uu = us.repeat_interleave(ds.n_items)
            ii = all_i.repeat(b)
            out[s:s + b] = torch.sigmoid(self.model(uu, ii)).view(b, ds.n_items).cpu().numpy()
        return out
