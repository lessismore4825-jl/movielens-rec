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
    epochs: int = 8
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
    search_users: int = 800         # 权重搜索所用的采样用户数
    search_step: float = 0.1        # 网格步长（权重之和固定为 1）


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
