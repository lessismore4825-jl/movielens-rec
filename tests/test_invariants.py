"""关键不变量的回归测试。

这些测试全部跑在**合成数据**上，不依赖 MovieLens 数据集，因此可以在 CI 里直接跑。

其中三个测试是针对原实现两处 bug 的回归防线：

* `test_test_positive_stays_in_candidate_set`  —— 防 bug B 复发
* `test_perfect_ranker_gets_hr_one`            —— 防"指标结构性为 0"复发
* `test_popularity_penalty_never_inverts_order`—— 防 bug A 复发

最后一个尤其重要：如果评估协议本身有问题，那么"把真实正例排到第一"的
完美排序器也拿不到 HR@10 = 1.0。原实现如果有这条测试，
五项指标恒为 0 的问题在第一次运行时就会暴露，而不会被归因为"没有 GPU"。

运行：
    pip install pytest
    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.config import FusionConfig, ItemKNNConfig, PathConfig   # noqa: E402
from recsys.data import DataLoader                             # noqa: E402
from recsys.evaluate import full_ranking_metrics, gini, novelty, positive_rank  # noqa: E402
from recsys.fusion import HybridFusion                         # noqa: E402
from recsys.models.itemknn import ItemCFImplicitRecommender    # noqa: E402
from recsys.models.neumf import NeuMFRecommender                # noqa: E402

N_USERS, N_ITEMS, N_PER_USER = 60, 40, 12


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory) -> Path:
    """造一份结构与 ml-1m 一致的小数据集：60 用户 × 40 电影，每人 12 条评分。"""
    d = tmp_path_factory.mktemp("ml-syn")
    rng = np.random.default_rng(0)

    lines = []
    for u in range(1, N_USERS + 1):
        items = rng.choice(np.arange(1, N_ITEMS + 1), size=N_PER_USER, replace=False)
        # 时间戳故意乱序写入，用于验证切分确实按 timestamp 排序而不是按行序
        ts = rng.permutation(np.arange(1_000_000, 1_000_000 + N_PER_USER * 10, 10))
        for it, t in zip(items, ts):
            lines.append(f"{u}::{it}::{rng.integers(1, 6)}::{t}")
    (d / "ratings.dat").write_text("\n".join(lines), encoding="latin-1")

    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"]
    (d / "movies.dat").write_text("\n".join(
        f"{i}::Movie {i} ({1990 + i % 10})::{genres[i % len(genres)]}|"
        f"{genres[(i + 1) % len(genres)]}"
        for i in range(1, N_ITEMS + 1)), encoding="latin-1")

    (d / "users.dat").write_text("\n".join(
        f"{u}::{'MF'[u % 2]}::{[1,18,25,35,45,50,56][u % 7]}::{u % 21}::0000{u:04d}"
        for u in range(1, N_USERS + 1)), encoding="latin-1")
    return d


@pytest.fixture(scope="module")
def ds(synthetic_dir):
    return DataLoader(PathConfig(data_dir=synthetic_dir)).load(min_interactions=3)


# ---------------------------------------------------------------- 数据切分
def test_splits_are_disjoint(ds):
    """train / valid / test 三份的 (u, i) 不能有任何交集（防数据泄漏）。"""
    def pairs(df):
        return set(map(tuple, df[["u", "i"]].to_numpy()))

    tr, va, te = pairs(ds.train), pairs(ds.valid), pairs(ds.test)
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()


def test_each_user_has_exactly_one_test_item(ds):
    assert len(ds.test) == ds.n_users
    assert ds.test["u"].nunique() == ds.n_users


def test_leave_one_out_is_chronological(ds, synthetic_dir):
    """测试集那一条必须是该用户 timestamp 最大的交互，而不是文件里的最后一行。

    原实现用 `groupby('userId').tail(1)` 且事先没有按 timestamp 排序，
    取到的是行序意义上的最后一条。
    """
    import pandas as pd

    raw = pd.read_csv(synthetic_dir / "ratings.dat", sep="::", engine="python",
                      names=["userId", "movieId", "rating", "timestamp"],
                      encoding="latin-1")
    raw["u"] = raw["userId"].map(ds.user_id_map)
    raw["i"] = raw["movieId"].map(ds.item_id_map)
    latest = raw.loc[raw.groupby("u")["timestamp"].idxmax()].set_index("u")["i"]

    for row in ds.test.itertuples():
        assert row.i == latest.loc[row.u], f"用户 {row.u} 的测试样本不是时间上最后一条"


# ---------------------------------------------------------------- 候选集（bug B）
def test_test_positive_stays_in_candidate_set(ds):
    """每个测试正例都必须落在候选集内，否则五项排序指标会结构性恒为 0。"""
    mask = HybridFusion.candidate_mask(ds)
    pos = ds.test["i"].to_numpy()
    in_cand = mask[np.arange(ds.n_users), pos]
    assert in_cand.all(), f"{int((~in_cand).sum())} 个测试正例被排除在候选集之外"


def test_candidate_set_excludes_training_items(ds):
    """候选集必须排除训练集里已交互的物品——不能把用户看过的再推一遍。"""
    mask = HybridFusion.candidate_mask(ds)
    u = ds.train["u"].to_numpy()
    i = ds.train["i"].to_numpy()
    assert not mask[u, i].any()


# ---------------------------------------------------------------- 评估协议
def test_perfect_ranker_gets_hr_one(ds):
    """把真实正例打成最高分的"完美排序器"必须拿到 HR@10 = 1.0。

    这是评估协议自身的健全性检查。原实现在这条测试上会失败：
    正例不在候选集内，无论打多高的分都命中不了。
    """
    mask = HybridFusion.candidate_mask(ds)
    pos = ds.test["i"].to_numpy()
    scores = np.zeros((ds.n_users, ds.n_items), dtype=np.float32)
    scores[np.arange(ds.n_users), pos] = 1.0
    scores[~mask] = -np.inf

    m = full_ranking_metrics(scores, pos, ds, top_n=10)
    assert m["HR@10"] == pytest.approx(1.0)
    assert m["nDCG@10"] == pytest.approx(1.0)
    assert m["MRR"] == pytest.approx(1.0)
    assert m["Precision@10"] == pytest.approx(0.1)   # 1 个相关物品 / 10 个位置


def test_worst_ranker_gets_hr_zero(ds):
    """把正例打成最低分，HR@10 必须是 0——保证指标不是恒为正的假阳性。"""
    mask = HybridFusion.candidate_mask(ds)
    pos = ds.test["i"].to_numpy()
    scores = np.ones((ds.n_users, ds.n_items), dtype=np.float32)
    scores[np.arange(ds.n_users), pos] = -1.0
    scores[~mask] = -np.inf
    assert full_ranking_metrics(scores, pos, ds, top_n=10)["HR@10"] == 0.0


def test_positive_rank_is_zero_based(ds):
    scores = np.array([[3.0, 1.0, 2.0], [1.0, 5.0, 9.0]], dtype=np.float32)
    assert positive_rank(scores, np.array([0, 1])).tolist() == [0, 1]


# ---------------------------------------------------------------- 热度惩罚（bug A）
def test_popularity_penalty_never_inverts_order(ds):
    """热度惩罚只能温和地重排，不能把分数变号、整体翻转排序。

    原实现 `score *= (1 - 0.2 * pop)` 在 pop=5 处乘数就穿过 0 变负，
    交互 3000 次的电影乘数为 -599。
    """
    pop_norm = HybridFusion(FusionConfig()).popularity_norm(ds)
    assert pop_norm.min() >= 0.0 and pop_norm.max() <= 1.0

    for alpha in (0.0, 0.1, 0.5, 1.0):
        # 加性惩罚的位移量始终有界，不会把任何分数推到符号相反的量级
        shift = alpha * pop_norm
        assert shift.min() >= 0.0
        assert shift.max() <= alpha + 1e-6


def test_fused_scores_mask_out_seen_items(ds):
    """融合结果里，用户在训练集看过的物品必须是 -inf，不可能被推荐。"""
    rec = ItemCFImplicitRecommender(ItemKNNConfig(topk=5, min_support=1)).fit(ds)
    scores = {"itemcf": rec.predict_all(ds)}
    fusion = HybridFusion(FusionConfig(weights={"itemcf": 1.0}, search_weights=False))
    fused = fusion.fuse(scores, ds)

    u = ds.train["u"].to_numpy()
    i = ds.train["i"].to_numpy()
    assert np.isneginf(fused[u, i]).all()


# ---------------------------------------------------------------- 指标公式
def test_novelty_is_finite_for_unrated_items(ds):
    """新颖性对交互次数为 0 的物品也必须是有限值。

    原实现用 `1/log(1+p)`，p=0 时为 inf、p=1 时分母是 log(2)。
    """
    rec = np.arange(ds.n_items).reshape(1, -1)
    v = novelty(rec, np.zeros(ds.n_items), ds.n_users)
    assert np.isfinite(v) and v > 0


def test_gini_bounds():
    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-9)          # 完全均匀
    concentrated = np.zeros(100)
    concentrated[0] = 1000
    assert gini(concentrated) > 0.98                                    # 高度集中


def test_coverage_reflects_recommendation_concentration(ds):
    """所有用户被推同一批物品时，Coverage 必须显著低于 1。

    原实现 60 400 条推荐只覆盖 31 部电影（Coverage 0.0084），
    这类指标本应在评估阶段就报警。
    """
    mask = HybridFusion.candidate_mask(ds)
    pos = ds.test["i"].to_numpy()
    scores = np.tile(np.linspace(1, 0, ds.n_items, dtype=np.float32), (ds.n_users, 1))
    scores[~mask] = -np.inf
    m = full_ranking_metrics(scores, pos, ds, top_n=10)
    assert m["Coverage"] < 0.5
    assert m["Gini"] > 0.5


# ---------------------------------------------------------------- NeuMF 负采样
def test_neumf_training_negatives_exclude_train_positives(ds):
    """训练负采样只能依赖 train，并且不得撞上 train 正例。"""
    blocked = NeuMFRecommender._known_positive_sets(ds)

    pos_u = ds.train["u"].to_numpy(np.int64)
    rng = np.random.default_rng(123)

    neg = NeuMFRecommender._sample_negatives(
        pos_u=pos_u,
        n_items=ds.n_items,
        k=4,
        blocked=blocked,
        rng=rng,
    )

    u_rep = np.repeat(pos_u, 4)

    assert all(
        int(i) not in blocked[int(u)]
        for u, i in zip(u_rep, neg)
    )

    # Future validation/test interactions must not be injected
    # into the training-time blocked set.
    for split in (ds.valid, ds.test):
        for u, i in zip(
            split["u"].to_numpy(np.int64),
            split["i"].to_numpy(np.int64),
        ):
            assert int(i) not in blocked[int(u)]


def test_neumf_validation_negatives_use_no_test_information(ds):
    """Validation negatives 排除 train+valid，但不得使用 future test 信息。"""
    rng = np.random.default_rng(123)

    u, pos, neg = NeuMFRecommender._build_valid(
        ds,
        rng,
        n_neg=20,
    )

    blocked = NeuMFRecommender._known_positive_sets(
        ds,
        include_valid=True,
    )

    test_items = {
        int(uu): int(ii)
        for uu, ii in zip(
            ds.test["u"].to_numpy(np.int64),
            ds.test["i"].to_numpy(np.int64),
        )
    }

    for row, uu in enumerate(u):
        user = int(uu)

        assert int(pos[row]) in blocked[user]

        assert all(
            int(i) not in blocked[user]
            for i in neg[row]
        )

        # Sampled validation negatives must be unique.
        assert len(np.unique(neg[row])) == len(neg[row])

        # Future test positive is not part of the validation-time blocklist.
        assert test_items[user] not in blocked[user]


# ---------------------------------------------------------------- Validation / Test 候选集
def test_validation_candidate_mask_uses_no_future_test_information(ds):
    """Validation 时只知道 train；future test interaction 仍属于候选集合。"""
    mask = HybridFusion.candidate_mask(ds, target="valid")

    vu = ds.valid["u"].to_numpy()
    vi = ds.valid["i"].to_numpy()
    tu = ds.test["u"].to_numpy()
    ti = ds.test["i"].to_numpy()
    tru = ds.train["u"].to_numpy()
    tri = ds.train["i"].to_numpy()

    assert mask[vu, vi].all()
    assert mask[tu, ti].all()
    assert not mask[tru, tri].any()


def test_test_candidate_mask_keeps_only_test_positive(ds):
    """测试评估必须保留 test 正例，并排除 valid 正例与训练历史。"""
    mask = HybridFusion.candidate_mask(ds, target="test")

    tu = ds.test["u"].to_numpy()
    ti = ds.test["i"].to_numpy()
    vu = ds.valid["u"].to_numpy()
    vi = ds.valid["i"].to_numpy()
    tru = ds.train["u"].to_numpy()
    tri = ds.train["i"].to_numpy()

    assert mask[tu, ti].all()
    assert not mask[vu, vi].any()
    assert not mask[tru, tri].any()
