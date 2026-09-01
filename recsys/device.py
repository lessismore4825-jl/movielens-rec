"""PyTorch 设备选择。

默认值是 ``"cpu"``，而不是"有什么用什么"。这是刻意的：

本项目的定位是一份可复现的离线基准，而 CPU 是唯一能在不同机器之间
比对数值的公共基准。MPS 与 CUDA 的浮点累加顺序与 CPU 不同，会改变
NeuMF 的逐轮验证 HR，进而改变早停点，最终让测试指标漂移约 ±0.002。

因此：
  * 汇报 Frozen 基准时用 ``--device cpu``（默认）；
  * 想在 Apple Silicon 上快速迭代时用 ``--device auto``。

两者的差异属于预期行为，不是 bug；README 的复现性章节对此有说明。
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

VALID = ("cpu", "auto", "mps", "cuda")


def resolve_device(name: str = "cpu") -> torch.device:
    """把命令行传入的设备名解析为具体的 torch.device。"""
    if name not in VALID:
        raise ValueError(f"未知设备 {name!r}，可选：{', '.join(VALID)}")

    if name == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
        if resolved != "cpu":
            logger.warning(
                "使用 %s 加速：浮点累加顺序与 CPU 不同，NeuMF 的早停点可能改变，"
                "测试指标会有约 ±0.002 的漂移。汇报基准数值请改用 --device cpu。",
                resolved,
            )
        logger.info("[device] auto -> %s", resolved)
        return torch.device(resolved)

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前环境没有可用的 CUDA 设备")
    if name == "mps" and not (getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()):
        raise RuntimeError("指定了 --device mps，但当前环境没有可用的 MPS 后端")

    return torch.device(name)
