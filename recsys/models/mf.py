"""Biased Matrix Factorization —— 对应原实现中的 surprise SVD。

为什么不再用 surprise
---------------------
1. surprise 已停止维护，在 Python 3.11+ / NumPy 2.x 上需要从源码编译，安装极易失败；
2. 它的 `predict()` 是逐条 Python 调用，原实现为 6040 位用户逐个候选调用了约
   2 200 万次，成为主要耗时来源之一；
3. 这里用 PyTorch 重写后，同样的模型可以按 mini-batch 向量化训练与推理，
   并且天然支持验证集早停——原实现的 SVD 是固定 30 轮，没有任何早停或验证。

模型
    r̂(u,i) = μ + b_u + b_i + p_u · q_i
损失
    MSE + L2（通过 AdamW 的 weight_decay 实现）
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from ..config import MFConfig
from ..data import Dataset
from .base import BaseRecommender

logger = logging.getLogger(__name__)


class _MFModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, global_mean: float):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        dot = (self.user_emb(u) * self.item_emb(i)).sum(-1)
        return self.global_mean + self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1) + dot


class MFRecommender(BaseRecommender):
    name = "mf"
    predicts_ratings = True

    def __init__(self, cfg: MFConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

    def fit(self, ds: Dataset, valid_cb=None) -> "MFRecommender":
        cfg = self.cfg
        dev = self.device
        gm = float(ds.train["rating"].mean())
        self.model = _MFModule(ds.n_users, ds.n_items, cfg.n_factors, gm).to(dev)

        u = torch.as_tensor(
            ds.train["u"].to_numpy(dtype=np.int64, copy=True)
        )
        i = torch.as_tensor(
            ds.train["i"].to_numpy(dtype=np.int64, copy=True)
        )
        r = torch.as_tensor(
            ds.train["rating"].to_numpy(dtype=np.float32, copy=True)
        )
        vu = torch.as_tensor(
            ds.valid["u"].to_numpy(dtype=np.int64, copy=True),
            device=dev,
        )
        vi = torch.as_tensor(
            ds.valid["i"].to_numpy(dtype=np.int64, copy=True),
            device=dev,
        )
        vr = torch.as_tensor(
            ds.valid["rating"].to_numpy(dtype=np.float32, copy=True),
            device=dev,
        )

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = len(u)
        best, best_state, bad = float("inf"), None, 0
        g = torch.Generator().manual_seed(0)

        for ep in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for s in range(0, n, cfg.batch_size):
                i_b = perm[s:s + cfg.batch_size]
                ub, ib, rb = u[i_b].to(dev), i[i_b].to(dev), r[i_b].to(dev)
                loss = nn.functional.mse_loss(self.model(ub, ib), rb)
                if cfg.reg > 0:
                    # 经典 MF 的 L2：只惩罚本 batch 用到的隐向量与偏置，
                    # 而不是像 AdamW 的 weight_decay 那样对全体参数无差别衰减。
                    m = self.model
                    loss = loss + cfg.reg * (
                        m.user_emb(ub).pow(2).sum() + m.item_emb(ib).pow(2).sum()
                        + m.user_bias(ub).pow(2).sum() + m.item_bias(ib).pow(2).sum()
                    ) / len(i_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(i_b)

            self.model.eval()
            with torch.no_grad():
                vrmse = torch.sqrt(nn.functional.mse_loss(
                    self.model(vu, vi).clamp(1, 5), vr)).item()
            logger.info("[mf] epoch %2d/%d  train_mse=%.4f  valid_rmse=%.4f",
                        ep + 1, cfg.epochs, total / n, vrmse)

            if vrmse < best - 1e-4:
                best, bad = vrmse, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    logger.info("[mf] 验证 RMSE 连续 %d 轮无提升，早停于第 %d 轮", bad, ep + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_valid_rmse_ = best
        return self

    @torch.no_grad()
    def predict_all(self, ds: Dataset) -> np.ndarray:
        self.model.eval()
        m = self.model
        P = m.user_emb.weight.detach()
        Q = m.item_emb.weight.detach()
        out = (P @ Q.T
               + m.user_bias.weight.detach()
               + m.item_bias.weight.detach().T
               + m.global_mean)
        return out.clamp(1.0, 5.0).cpu().numpy().astype(np.float32)
