"""全局配置。所有超参数集中在此，便于复现与消融实验。"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------- 路径配置
@dataclass
class PathConfig:
    """数据与输出路径。默认相对当前工作目录，可用 --data-dir / --out-dir 覆盖。"""

    data_dir: Path = Path("data/ml-1m")
    out_dir: Path = Path("outputs")

    @property
    def ratings(self) -> Path:
        return self.data_dir / "ratings.dat"

    @property
    def movies(self) -> Path:
        return self.data_dir / "movies.dat"

    @property
    def users(self) -> Path:
        return self.data_dir / "users.dat"

    def validate(self) -> None:
        missing = [p for p in (self.ratings, self.movies, self.users) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "找不到数据文件：\n  "
                + "\n  ".join(str(p) for p in missing)
                + f"\n请用 --data-dir 指定 ml-1m 目录（当前为 {self.data_dir.resolve()}）"
            )


# ---------------------------------------------------------------- 模型超参
@dataclass
class ItemKNNConfig:
    topk: int = 40              # 每个物品保留的近邻数
    shrinkage: float = 20.0     # 共评人数收缩系数，抑制低支持度的虚高相似度
    min_support: int = 5        # 共同评分人数下限


@dataclass
class MFConfig:
    """Biased Matrix Factorization（对应原代码的 SVD）。"""

    n_factors: int = 32
    epochs: int = 40
    batch_size: int = 8192
    lr: float = 4e-3
    reg: float = 0.05           # 经典 MF 的 L2 系数（只作用于 batch 内的隐向量）
    weight_decay: float = 0.0   # AdamW 的全局衰减，默认关闭，正则由 reg 承担
    patience: int = 4


@dataclass
class NeuMFConfig:
    """NeuMF：隐式反馈 + 负采样 + BCE，对应 He et al. 2017 的标准训练目标。"""

    mf_dim: int = 32
    mlp_dim: int = 32
    mlp_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.2
    n_negatives: int = 4        # 每个正样本配几个负样本
    epochs: int = 25
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 2


@dataclass
class ContentConfig:
    tfidf_max_features: int = 20000
    use_sbert: bool = False     # 需能访问 HuggingFace 才可开启
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    sbert_weight: float = 0.5   # SBERT 与 TF-IDF 的融合比例


@dataclass
class FusionConfig:
    """混合融合层。"""

    weights: Dict[str, float] = field(
        default_factory=lambda: {"itemknn": 0.1, "itemcf": 0.3, "mf": 0.1,
                                 "neumf": 0.4, "content": 0.1}
    )
    normalize: str = "zscore"       # zscore | minmax | rank
    popularity_alpha: float = 0.10  # 热度惩罚强度，取值 [0, 1]；0 表示不惩罚
    search_weights: bool = True     # 是否在验证集上网格搜索权重
    search_users: int = 0           # 参与权重搜索的验证用户数；0 = 全部
    search_step: float = 0.1        # 网格步长（权重之和固定为 1）
    search_objective: str = "full"  # full | sampled，见下方说明


# ---------------------------------------------------------------------------
# 关于 search_users 与 search_objective 的默认值
#
# 早期版本用 800 位验证用户 + 全候选 nDCG@10 作为权重搜索目标。
# 全候选协议下验证 nDCG@10 约 0.04，800 位用户合计只有约 48 次命中，
# 用这个稀疏信号从 1001 个权重组合里挑最优，容易产生较大的选择方差。
#
# 受控实验（experiments/weight_search_stability.py，固定同一批打分矩阵、
# 只重采验证用户子集 12 次）显示：模型参数一字未变，选出的 NeuMF 权重
# 却在 0.2 ~ 1.0 之间摆动，测试 HR@10 落在 0.0740 ~ 0.0810。
#
# 受控实验表明，仅验证用户子采样本身就足以造成明显的权重选择不稳定。
# 因此 Frozen V4 默认使用全部验证用户，以减少这一已识别的选择方差来源。
#
#   800 位用户  / 全候选目标   HR@10 = 0.0777 ± 0.0026   (+2.9% vs 最强单模型)
#   全部用户    / 全候选目标   HR@10 = 0.0810            (+7.2%)
#   全部用户    / 采样负例目标 HR@10 = 0.0823            (+9.0%)
#
# search_objective 默认为 "full"：搜索目标与最终汇报的主协议一致，
# 是更保守、也更容易辩护的选择。
# "sampled" 用 1 正 + 99 负的 nDCG@10 作为低方差代理目标——它的信号密度
# 高一个数量级（HR≈0.7 而非 0.04），实测还能再好一些，但选择指标与
# 汇报指标不同，使用时需在结论中写明。


@dataclass
class EvalConfig:
    top_n: int = 10
    n_negatives: int = 99       # 留一法评估的负采样个数（NeuMF 论文协议）


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    itemknn: ItemKNNConfig = field(default_factory=ItemKNNConfig)
    mf: MFConfig = field(default_factory=MFConfig)
    neumf: NeuMFConfig = field(default_factory=NeuMFConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: str = "cpu"     # cpu | auto | mps | cuda，见 recsys/device.py

    def to_dict(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


def set_seed(seed: int) -> None:
    """固定所有随机源，保证结果可复现（原代码完全没有设种子）。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
