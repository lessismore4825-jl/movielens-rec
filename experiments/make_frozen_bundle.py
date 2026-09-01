"""把一次多种子运行的产物打包成可校验的 Frozen 证据包。

之前的 Frozen 包是手工拼的，SHA256SUMS.txt 也是手工维护——结果 V3 里
`tie_audit.txt` 被 .gitignore 挡住却仍留在校验清单里，任何人 clone 之后
跑 `sha256sum -c` 都会失败。用脚本生成可以避免这类不一致。

本脚本做四件事：

1. 收集 ``--run-dir`` 下的产物文件；
2. 运行一次测试套件并把输出存为 ``pytest.txt``；
3. 生成 ``README.md``（说明这个包是什么、怎么复现）；
4. 计算 ``SHA256SUMS.txt``，并**校验清单中的每个文件都确实存在且未被
   .gitignore 排除**。

用法：
    python -m experiments.make_frozen_bundle \
        --run-dir results/frozen_v4 --version v4
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

BUNDLE_FILES = [
    "metrics_seed42.json",
    "recommendations_sample_seed42.csv",
    "multiseed_raw.csv",
    "multiseed_summary.csv",
    "fusion_weights.csv",
    "training_details.csv",
    "paired_hybrid_vs_best_single.csv",
    "ablation_popularity.csv",
    "ablation_bugs.csv",
    "environment.txt",
    "pytest.txt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_ignored(repo_root: Path, path: Path) -> bool:
    """检查文件是否会被 .gitignore 排除——防止再次出现"清单里有、仓库里没有"。"""
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(path)],
                           cwd=repo_root, capture_output=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--version", default="v4")
    p.add_argument("--seeds", default="42, 43, 44, 45, 46")
    p.add_argument("--canonical-seed", type=int, default=42)
    p.add_argument("--skip-pytest", action="store_true")
    a = p.parse_args()

    d: Path = a.run_dir
    if not d.is_dir():
        print(f"找不到目录 {d}", file=sys.stderr)
        return 1
    repo_root = Path(__file__).resolve().parents[1]

    # ---- 1. 跑测试并存档 ----
    if not a.skip_pytest:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            "-p", "no:cacheprovider"],
                           cwd=repo_root, capture_output=True, text=True)
        (d / "pytest.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        print(f"pytest: {r.stdout.strip().splitlines()[-1] if r.stdout else '见 pytest.txt'}")
        if r.returncode != 0:
            print("测试未全部通过，Frozen 包不应发布", file=sys.stderr)
            return 1

    present = [f for f in BUNDLE_FILES if (d / f).exists()]
    missing = [f for f in BUNDLE_FILES if not (d / f).exists()]

    if missing:
        print(
            "Frozen bundle is incomplete; required artifacts are missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    # ---- 2. 生成 README ----
    readme = f"""# Frozen {a.version.upper()}

本目录是一次完整多种子运行的冻结产物，用于支撑仓库根目录 README 中的所有数字。

## 运行配置

- 数据集：MovieLens-1M（未随仓库分发，见 `data/README.md`）
- 切分：按时间的 leave-one-out（train / validation / test）
- 种子：{a.seeds}
- 正式展示种子：{a.canonical_seed}
- 主协议：全候选 Top-10 排序
- 并列规则：分数降序，完全同分时物品内部索引升序
- 设备：CPU（保证跨平台数值可比，见根 README 的复现性说明）

## 复现

```bash
python -m experiments.multiseed \\
    --seeds {a.seeds.replace(',', '')} \\
    --out-dir results/frozen_{a.version}

python -m experiments.make_frozen_bundle \\
    --run-dir results/frozen_{a.version} --version {a.version}
```

## 文件

""" + "\n".join(f"- `{f}`" for f in present) + """
- `SHA256SUMS.txt`

## 校验

```bash
cd """ + f"results/frozen_{a.version}" + """
sha256sum -c SHA256SUMS.txt
```
"""
    (d / "README.md").write_text(readme, encoding="utf-8")

    # ---- 3. 生成并自检 SHA256SUMS ----
    listed = ["README.md"] + present
    lines, problems = [], []
    for f in listed:
        path = d / f
        if not path.exists():
            problems.append(f"{f}：文件不存在")
            continue
        if git_ignored(repo_root, path):
            problems.append(f"{f}：会被 .gitignore 排除，clone 后校验必然失败")
        lines.append(f"{sha256(path)}  {f}")
    (d / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if problems:
        print("\n清单自检发现问题：", file=sys.stderr)
        for msg in problems:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print(f"\nFrozen {a.version.upper()} 已打包：{len(lines)} 个文件，清单自检通过")
    print(f"目录：{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
