"""多种子鲁棒性实验：把整条流水线在若干随机种子下各跑一遍并汇总。

为什么需要它
------------
单次运行的 HR@10 差异可能受到随机性的显著影响。要主张"融合系统优于最强单模型"，
必须给出跨种子的均值与标准差，以及配对差值（同一种子下 hybrid 减 best-single）。

种子会同时影响两件事：
  1. NeuMF / MF 的参数初始化与负采样
  2. 评估阶段采样负例协议所用的负样本

（自 Frozen V4 起，权重搜索默认使用全部验证用户，因此种子不再影响
  搜索所用的用户子集——受控实验表明，验证用户子采样本身足以造成明显的权重选择方差，但不能解释 Frozen V3 的全部方差。）

用法
----
    python -m experiments.multiseed --seeds 42 43 44 45 46 --out-dir results/frozen_v4
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import Config, PathConfig, set_seed
from recsys.pipeline import Pipeline

logger = logging.getLogger(__name__)

REPORT_METRICS = ["HR@10", "nDCG@10", "MRR", "Coverage", "Gini",
                  "HR@10(sampled)", "nDCG@10(sampled)", "AUC(sampled)"]


def environment_report() -> str:
    import scipy
    import sklearn
    import torch

    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return "\n".join([
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"Pandas: {pd.__version__}",
        f"SciPy: {scipy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"MPS available: {mps}",
    ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/multiseed"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--canonical-seed", type=int, default=42,
                   help="用哪个种子的产物作为对外展示的正式结果")
    p.add_argument("--neumf-epochs", type=int, default=25)
    p.add_argument("--search-objective", choices=["full", "sampled"], default="full")
    p.add_argument("--search-users", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows, weight_rows, training_rows = [], [], []
    t_start = time.time()

    for seed in a.seeds:
        logger.info("═" * 30 + f"  seed = {seed}  " + "═" * 30)
        cfg = Config(paths=PathConfig(data_dir=a.data_dir,
                                      out_dir=a.out_dir / f"seed{seed}"),
                     seed=seed)
        cfg.neumf.epochs = a.neumf_epochs
        cfg.fusion.search_objective = a.search_objective
        cfg.fusion.search_users = a.search_users
        cfg.device = a.device
        cfg.paths.out_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)

        pipe = Pipeline(cfg)
        pipe.load()
        pipe.build_models()
        pipe.train_and_score()
        pipe.fuse()
        res = pipe.evaluate()

        if seed == a.canonical_seed:
            df_pop = pipe.ablate_popularity()
            df_pop.to_csv(a.out_dir / "ablation_popularity.csv",
                          index=False, encoding="utf-8-sig")
            from experiments.legacy_bug_ablation import run_bug_ablation
            run_bug_ablation({m.name: pipe.scores[m.name] for m in pipe.models},
                             pipe.ds, cfg.eval.top_n, seed).to_csv(
                a.out_dir / "ablation_bugs.csv", index=False, encoding="utf-8-sig")
            files = pipe.export(top_n=cfg.eval.top_n)
            rec = pd.read_csv(files["recommendations"])
            rec.head(200).to_csv(a.out_dir / f"recommendations_sample_seed{seed}.csv",
                                 index=False, encoding="utf-8-sig")
            (a.out_dir / f"metrics_seed{seed}.json").write_text(
                json.dumps({"config": cfg.to_dict(), "results": res},
                           ensure_ascii=False, indent=2, default=float),
                encoding="utf-8")

        for name, m in list(res["per_model"].items()) + [("hybrid", res["hybrid"])]:
            rows.append({"seed": seed, "model": name,
                         **{k: m[k] for k in REPORT_METRICS if k in m}})
        weight_rows.append({"seed": seed,
                            **{k: float(v) for k, v in res["weights"].items()}})
        training_rows.append({"seed": seed, **{
            f"{mdl}.{k}": v for mdl, d in res["training"].items()
            if isinstance(d, dict) for k, v in d.items() if k != "valid_hr_history"}})

    raw = pd.DataFrame(rows)
    raw.to_csv(a.out_dir / "multiseed_raw.csv", index=False, encoding="utf-8-sig")
    summary = raw.groupby("model")[REPORT_METRICS].agg(["mean", "std"])
    summary.to_csv(a.out_dir / "multiseed_summary.csv", encoding="utf-8-sig")
    pd.DataFrame(weight_rows).to_csv(a.out_dir / "fusion_weights.csv",
                                     index=False, encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(a.out_dir / "training_details.csv",
                                       index=False, encoding="utf-8-sig")
    (a.out_dir / "environment.txt").write_text(environment_report(), encoding="utf-8")

    # ---- 配对比较：同一种子下 hybrid 减去当次最强单模型 ----
    singles = raw[raw["model"] != "hybrid"]
    best_single = (singles.loc[singles.groupby("seed")["HR@10"].idxmax()]
                   .set_index("seed"))
    hyb = raw[raw["model"] == "hybrid"].set_index("seed")
    paired = pd.DataFrame({
        "best_single_model": best_single["model"],
        "best_single_HR@10": best_single["HR@10"],
        "hybrid_HR@10": hyb["HR@10"],
        "diff_HR@10": hyb["HR@10"] - best_single["HR@10"],
        "best_single_nDCG@10": best_single["nDCG@10"],
        "hybrid_nDCG@10": hyb["nDCG@10"],
        "diff_nDCG@10": hyb["nDCG@10"] - best_single["nDCG@10"],
    })
    paired.to_csv(a.out_dir / "paired_hybrid_vs_best_single.csv", encoding="utf-8-sig")

    print("\n════ 跨种子汇总（均值 ± 标准差）════")
    print(summary[["HR@10", "nDCG@10", "Coverage"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n════ 配对比较：hybrid vs 当次最强单模型 ════")
    print(paired.to_string(float_format=lambda x: f"{x:.4f}"))
    d = paired["diff_HR@10"]
    wins = int((d > 0).sum())
    print(f"\nHR@10 配对差值：均值 {d.mean():+.4f}，标准差 {d.std():.4f}，"
          f"{wins}/{len(d)} 个种子上融合更优")
    print(f"\n全部产物已写入 {a.out_dir}，总耗时 {(time.time()-t_start)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
