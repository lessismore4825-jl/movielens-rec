#!/usr/bin/env python3
"""MovieLens-1M 混合推荐系统 —— 命令行入口。

用法
----
    python run.py --data-dir /path/to/ml-1m --out-dir outputs

常用开关
    --no-weight-search      跳过权重网格搜索（用配置里的固定权重）
    --pop-alpha 0.0         关闭热度惩罚
    --ablation              额外跑一遍热度惩罚强度消融（含原实现错误公式的对照）
    --neumf-epochs 8        调整 NeuMF 训练轮数
    --quick                 小规模冒烟测试，几十秒内跑完全流程
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from recsys.config import Config, PathConfig, set_seed
from recsys.evaluate import format_metrics
from recsys.pipeline import Pipeline


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MovieLens 混合推荐系统")
    p.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"),
                   help="ml-1m 目录（含 ratings.dat / movies.dat / users.dat）")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--pop-alpha", type=float, default=None, help="热度惩罚强度 [0,1]")
    p.add_argument("--normalize", choices=["zscore", "minmax", "rank"], default=None)
    p.add_argument("--no-weight-search", action="store_true")
    p.add_argument("--search-users", type=int, default=None,
                   help="参与权重搜索的验证用户数；0 表示全部（默认）")
    p.add_argument("--search-objective", choices=["full", "sampled"], default=None,
                   help="权重搜索目标：full=全候选 nDCG@10（默认，与主协议一致）；"
                        "sampled=1正+99负的低方差代理目标")
    p.add_argument("--device", choices=["cpu", "auto", "mps", "cuda"], default="cpu",
                   help="PyTorch 设备。默认 cpu 以保证跨平台数值可比；"
                        "Apple Silicon 上用 auto 更快，但 NeuMF 早停点可能改变")
    p.add_argument("--neumf-epochs", type=int, default=None)
    p.add_argument("--mf-epochs", type=int, default=None)
    p.add_argument("--use-sbert", action="store_true", help="启用 SBERT 语义特征（需联网）")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--bug-ablation", action="store_true",
                   help="复现原实现的两处 bug，做 2×2 受控对照")
    p.add_argument("--quick", action="store_true", help="冒烟测试：极少轮次 + 不搜权重")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def build_config(a: argparse.Namespace) -> Config:
    cfg = Config(paths=PathConfig(data_dir=a.data_dir, out_dir=a.out_dir), seed=a.seed)
    cfg.eval.top_n = a.top_n
    if a.pop_alpha is not None:
        cfg.fusion.popularity_alpha = a.pop_alpha
    if a.normalize:
        cfg.fusion.normalize = a.normalize
    if a.no_weight_search:
        cfg.fusion.search_weights = False
    if a.search_users is not None:
        cfg.fusion.search_users = a.search_users
    if a.search_objective:
        cfg.fusion.search_objective = a.search_objective
    cfg.device = a.device
    if a.neumf_epochs is not None:
        cfg.neumf.epochs = a.neumf_epochs
    if a.mf_epochs is not None:
        cfg.mf.epochs = a.mf_epochs
    cfg.content.use_sbert = a.use_sbert
    if a.quick:
        cfg.neumf.epochs = 1
        cfg.mf.epochs = 2
        cfg.fusion.search_weights = False
    return cfg


def main(argv=None) -> int:
    a = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, a.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    cfg = build_config(a)
    set_seed(cfg.seed)

    # Ensure all experiment artifacts can be written before evaluation/export.
    cfg.paths.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = Pipeline(cfg)
    pipe.load()
    pipe.build_models()
    pipe.train_and_score()
    pipe.fuse()
    results = pipe.evaluate()

    if a.ablation:
        df = pipe.ablate_popularity()
        cols = ["设置", f"HR@{cfg.eval.top_n}", f"nDCG@{cfg.eval.top_n}",
                "Coverage", "CoveredItems", "Novelty", "Gini"]
        print("\n热度惩罚消融（全候选排序协议）")
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        df.to_csv(cfg.paths.out_dir / "ablation_popularity.csv",
                  index=False, encoding="utf-8-sig")

    if a.bug_ablation:
        from experiments.legacy_bug_ablation import run_bug_ablation
        df = run_bug_ablation({m.name: pipe.scores[m.name] for m in pipe.models},
                              pipe.ds, cfg.eval.top_n, cfg.seed)
        print("\n原实现 bug 的 2×2 受控对照（同一份数据切分、同一批模型）")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        df.to_csv(cfg.paths.out_dir / "ablation_bugs.csv",
                  index=False, encoding="utf-8-sig")

    files = pipe.export(top_n=cfg.eval.top_n)
    total = time.time() - t0

    print("\n" + format_metrics("最终：混合推荐系统（全候选排序）", results["hybrid"]))
    print(f"\n最优融合权重: {results['weights']}")
    for k, v in files.items():
        print(f"输出 {k}: {v}")
    print(f"总耗时 {total/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
