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
        self.device = torch.device(device)
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
        """为每个正样本采 k 个不属于该用户已知正交互集合的负样本。"""
        m = len(pos_u) * k
        u_rep = np.repeat(pos_u, k)
        neg = rng.integers(0, n_items, size=m, dtype=np.int64)

        while True:
            bad = np.fromiter(
                (int(i) in blocked[int(u)] for u, i in zip(u_rep, neg)),
                dtype=bool,
                count=m,
            )

            if not bad.any():
                break

            neg[bad] = rng.integers(
                0,
                n_items,
                size=int(bad.sum()),
                dtype=np.int64,
            )

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

            if hr > best + 1e-4:
                best, bad = hr, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    logger.info("[neumf] 验证 HR 连续 %d 轮无提升，早停于第 %d 轮", bad, ep + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_valid_hr_ = best
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
        # 正例位于第 0 列，其排名 = 严格高于它的负例个数
        rank = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
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
