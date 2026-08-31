"""端到端流水线：训练 → 融合 → 评估 → 导出。"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import evaluate as ev
from .config import Config, set_seed
from .data import DataLoader, Dataset
from .fusion import HybridFusion, normalize_scores
from .models.base import BaseRecommender, PopularityRecommender, RandomRecommender
from .models.content import ContentRecommender
from .models.itemknn import ItemCFImplicitRecommender, ItemKNNRecommender
from .models.mf import MFRecommender
from .models.neumf import NeuMFRecommender

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------ 各阶段
    def load(self) -> Dataset:
        t = time.time()
        self.ds = DataLoader(self.cfg.paths).load()
        self.timings["load"] = time.time() - t
        return self.ds

    def build_models(self) -> List[BaseRecommender]:
        c = self.cfg
        self.models = [
            ItemKNNRecommender(c.itemknn),
            ItemCFImplicitRecommender(c.itemknn),
            MFRecommender(c.mf),
            NeuMFRecommender(c.neumf, seed=c.seed),
            ContentRecommender(c.content),
        ]
        self.baselines = [PopularityRecommender(), RandomRecommender(seed=c.seed)]
        return self.models

    def train_and_score(self) -> Dict[str, np.ndarray]:
        """训练全部模型并产出打分矩阵。"""
        self.scores: Dict[str, np.ndarray] = {}
        for m in self.models + self.baselines:
            m.timed_fit(self.ds)
            t = time.time()
            self.scores[m.name] = m.predict_all(self.ds)
            self.timings[f"predict:{m.name}"] = time.time() - t
            self.timings[f"fit:{m.name}"] = getattr(m, "fit_seconds", 0.0)
            logger.info("[%s] 打分矩阵 %s，推理耗时 %.1fs",
                        m.name, self.scores[m.name].shape, self.timings[f"predict:{m.name}"])
        return self.scores

    # ------------------------------------------------------------ 融合
    def fuse(self) -> np.ndarray:
        ds, cfg = self.ds, self.cfg
        self.fusion = HybridFusion(cfg.fusion)
        self.cand_mask = HybridFusion.candidate_mask(ds, target="test")

        model_scores = {m.name: self.scores[m.name] for m in self.models}

        if cfg.fusion.search_weights:
            t = time.time()

            def objective(fused_sub: np.ndarray, targets: np.ndarray) -> float:
                rank = ev.positive_rank(fused_sub, targets)
                n = cfg.eval.top_n
                return float(np.where(rank < n, 1.0 / np.log2(rank + 2.0), 0.0).mean())

            self.fusion.search_weights(model_scores, ds, objective, self.rng)
            self.timings["weight_search"] = time.time() - t

        t = time.time()
        self.fused = self.fusion.fuse(model_scores, ds, self.cand_mask)
        self.timings["fuse"] = time.time() - t
        return self.fused

    # ------------------------------------------------------------ 评估
    def _mask_single(self, name: str) -> np.ndarray:
        """把单个模型的分数限制到候选集内，用于与混合系统同口径比较。"""
        S = normalize_scores(self.scores[name], self.cand_mask, self.cfg.fusion.normalize)
        S = S.copy()
        S[~self.cand_mask] = -np.inf
        return S

    def evaluate(self) -> dict:
        ds, cfg = self.ds, self.cfg
        pos = ds.test["i"].to_numpy()
        top_n = cfg.eval.top_n

        # 断言：修复后的候选集必须包含全部测试正例
        in_cand = self.cand_mask[np.arange(ds.n_users), pos]
        assert in_cand.all(), (
            f"仍有 {int((~in_cand).sum())} 个测试正例不在候选集内——评估协议有误")
        logger.info("候选集自检通过：%d 个测试正例全部在候选集内", len(pos))

        results: dict = {"per_model": {}, "hybrid": {}}

        for name in list(self.scores):
            S = self._mask_single(name)
            m = ev.full_ranking_metrics(S, pos, ds, top_n)
            m.update(ev.sampled_ranking_metrics(
                S, pos, ds, np.random.default_rng(cfg.seed), top_n, cfg.eval.n_negatives))
            results["per_model"][name] = m
            logger.info("%s", ev.format_metrics(f"单模型 {name}", m))
            del S

        h = ev.full_ranking_metrics(self.fused, pos, ds, top_n)
        h.update(ev.sampled_ranking_metrics(
            self.fused, pos, ds, np.random.default_rng(cfg.seed), top_n, cfg.eval.n_negatives))
        results["hybrid"] = h
        logger.info("%s", ev.format_metrics("混合推荐系统", h))

        results["regression"] = ev.evaluate_rating_models(self.models, ds, self.scores)
        for k, v in results["regression"].items():
            logger.info("回归指标 %-22s MAE=%.4f RMSE=%.4f NMAE=%.4f NRMSE=%.4f",
                        k, v["MAE"], v["RMSE"], v["NMAE"], v["NRMSE"])

        results["weights"] = self.fusion.weights
        results["timings"] = self.timings
        self.results = results
        return results

    # ------------------------------------------------------------ 消融
    def ablate_popularity(self, alphas=(0.0, 0.05, 0.1, 0.2, 0.4)) -> pd.DataFrame:
        """热度惩罚强度消融，并复现原实现的错误公式作为对照。"""
        ds, cfg = self.ds, self.cfg
        pos = ds.test["i"].to_numpy()
        model_scores = {m.name: self.scores[m.name] for m in self.models}
        rows = []

        for a in alphas:
            f = self.fusion.fuse(model_scores, ds, self.cand_mask, alpha=a)
            m = ev.full_ranking_metrics(f, pos, ds, cfg.eval.top_n)
            rows.append({"设置": f"修正公式 alpha={a}", **m})
            del f

        # ---- 复现原实现：score *= (1 - 0.2 * 原始交互次数) ----
        base = self.fusion.fuse(model_scores, ds, self.cand_mask, alpha=0.0)
        buggy = np.where(self.cand_mask, base, 0.0)
        buggy = buggy * (1.0 - 0.2 * ds.item_popularity[None, :].astype(np.float32))
        buggy[~self.cand_mask] = -np.inf
        m = ev.full_ranking_metrics(buggy, pos, ds, cfg.eval.top_n)
        rows.append({"设置": "原实现公式 score*(1-0.2*pop)", **m})
        del base, buggy

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------ 导出
    def export(self, top_n: int = 10) -> Dict[str, Path]:
        ds, out_dir = self.ds, self.cfg.paths.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        files: Dict[str, Path] = {}

        rec = ev.top_n_items(self.fused, top_n)
        rows = ds.n_users
        inv_user = {v: k for k, v in ds.user_id_map.items()}
        df = pd.DataFrame({
            "userId": np.repeat([inv_user[u] for u in range(rows)], top_n),
            "rank": np.tile(np.arange(1, top_n + 1), rows),
            "movieId": ds.movies["movieId"].to_numpy()[rec.ravel()],
            "title": ds.movies["title"].to_numpy()[rec.ravel()],
            "genres": ds.movies["genres"].to_numpy()[rec.ravel()],
            "score": self.fused[np.repeat(np.arange(rows), top_n), rec.ravel()],
        })
        p = out_dir / f"recommendations_{stamp}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        files["recommendations"] = p

        p = out_dir / f"metrics_{stamp}.json"
        p.write_text(json.dumps(
            {"config": self.cfg.to_dict(), "results": self.results},
            ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        files["metrics"] = p
        return files
