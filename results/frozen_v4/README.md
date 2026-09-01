# Frozen V4

本目录是一次完整多种子运行的冻结产物，用于支撑仓库根目录 README 中的所有数字。

## 运行配置

- 数据集：MovieLens-1M（未随仓库分发，见 `data/README.md`）
- 切分：按时间的 leave-one-out（train / validation / test）
- 种子：42, 43, 44, 45, 46
- 正式展示种子：42
- 主协议：全候选 Top-10 排序
- 并列规则：分数降序，完全同分时物品内部索引升序
- 设备：CPU（保证跨平台数值可比，见根 README 的复现性说明）

## 复现

```bash
python -m experiments.multiseed \
    --seeds 42 43 44 45 46 \
    --out-dir results/frozen_v4

python -m experiments.make_frozen_bundle \
    --run-dir results/frozen_v4 --version v4
```

## 文件

- `metrics_seed42.json`
- `recommendations_sample_seed42.csv`
- `multiseed_raw.csv`
- `multiseed_summary.csv`
- `fusion_weights.csv`
- `training_details.csv`
- `paired_hybrid_vs_best_single.csv`
- `ablation_popularity.csv`
- `ablation_bugs.csv`
- `environment.txt`
- `pytest.txt`
- `SHA256SUMS.txt`

## 校验

```bash
cd results/frozen_v4
sha256sum -c SHA256SUMS.txt
```
