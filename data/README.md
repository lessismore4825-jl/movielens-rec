# 数据集放置说明

本仓库**不包含** MovieLens 数据集。GroupLens 的使用许可明确规定：

> The user may not redistribute the data without separate permission.
> （未经单独许可，使用者不得再分发本数据集。）

所以 `ratings.dat` / `movies.dat` / `users.dat` 已被 `.gitignore` 排除，
请自行下载后放到本目录：

```
data/
└── ml-1m/
    ├── ratings.dat
    ├── movies.dat
    ├── users.dat
    └── README
```

下载地址：<https://files.grouplens.org/datasets/movielens/ml-1m.zip>

```bash
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/
```

放好之后即可直接运行：

```bash
python run.py                       # 默认读 data/ml-1m
python run.py --data-dir /其他/路径/ml-1m
```

## 引用

使用本数据集的论文需引用：

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. ACM Transactions on Interactive Intelligent Systems
> (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
> DOI: http://dx.doi.org/10.1145/2827872
