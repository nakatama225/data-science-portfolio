"""`modeling` のテスト。

3段階に分けている。

1. 探索範囲の定義（`specs`）… 機械学習ライブラリを使わずに検査できる
2. 学習の流れ（`training`）… 標準化が正しい位置に入っているかを確かめる
3. 評価（`evaluation`）… 指標の計算が意図どおりかを、手で数えた値と突き合わせる
"""

from __future__ import annotations

import numpy as np
import pytest

from spacex_capstone.modeling.evaluation import (
    ConfusionCounts,
    comparison_table,
    majority_class_accuracy,
    score_model,
)
from spacex_capstone.modeling.specs import ModelSpec, total_fit_count
from spacex_capstone.modeling.training import (
    CLASSIFIER_STEP,
    build_pipeline,
    default_model_specs,
    prefixed_param_grid,
    split_train_test,
    train_with_grid_search,
)


class DummyEstimator:
    """機械学習ライブラリを持ち込まずに `ModelSpec` を検査するための置き石。"""


class Test探索範囲の定義:
    def test_組み合わせ数は候補数の積(self) -> None:
        spec = ModelSpec(
            name="テスト",
            estimator_factory=DummyEstimator,
            param_grid={"a": [1, 2, 3], "b": ["x", "y"]},
        )

        assert spec.combination_count == 6

    def test_名前が空なら作れない(self) -> None:
        with pytest.raises(ValueError, match="モデル名が空"):
            ModelSpec(
                name="  ", estimator_factory=DummyEstimator, param_grid={"a": [1]}
            )

    def test_候補が空の設定値があれば作れない(self) -> None:
        """空のリストを渡すと、探索が1件も行われないまま「完了」する。

        学習を始める前に弾く。
        """
        with pytest.raises(ValueError, match="候補が空"):
            ModelSpec(
                name="テスト", estimator_factory=DummyEstimator, param_grid={"a": []}
            )

    def test_設定値が1つも無ければ作れない(self) -> None:
        with pytest.raises(ValueError, match="試す設定値が1つも無い"):
            ModelSpec(name="テスト", estimator_factory=DummyEstimator, param_grid={})

    def test_作った後は書き換えられない(self) -> None:
        """探索の途中で範囲が変わると、どの範囲で得た結果か分からなくなる。"""
        spec = ModelSpec(
            name="テスト", estimator_factory=DummyEstimator, param_grid={"a": [1]}
        )

        with pytest.raises(AttributeError):
            spec.name = "別の名前"  # type: ignore[misc]


class Test学習回数の見積もり:
    def test_全モデルの組み合わせ数に分割数を掛ける(self) -> None:
        specs = [
            ModelSpec("A", DummyEstimator, {"a": [1, 2]}),
            ModelSpec("B", DummyEstimator, {"b": [1, 2, 3]}),
        ]

        assert total_fit_count(specs, cv_folds=10) == (2 + 3) * 10

    def test_分割数が1では検証データが残らない(self) -> None:
        with pytest.raises(ValueError, match="2以上"):
            total_fit_count([ModelSpec("A", DummyEstimator, {"a": [1]})], cv_folds=1)


class Test既定のモデル定義:
    def test_4種類のモデルを比較する(self) -> None:
        assert len(default_model_specs()) == 4

    def test_すべて名前が異なる(self) -> None:
        """名前が重なると、結果の一覧でどちらの成績か区別できない。"""
        names = [spec.name for spec in default_model_specs()]

        assert len(set(names)) == len(names)

    def test_探索の規模を実行前に把握できる(self) -> None:
        total = total_fit_count(default_model_specs(), cv_folds=10)

        assert total > 0
        # 規模が桁違いに増えたら気づけるよう、上限を置いておく。
        assert total < 100_000


class Test学習の流れ:
    def test_標準化がモデルの手前に入る(self) -> None:
        """標準化を流れの中に入れることで、分割ごとの訓練データだけで
        基準値が計算される。外に出すと検証データの情報が混ざる。
        """
        spec = default_model_specs()[0]

        pipeline = build_pipeline(spec)

        assert [name for name, _ in pipeline.steps] == ["scaler", CLASSIFIER_STEP]

    def test_設定値の名前に流れの位置が付く(self) -> None:
        spec = ModelSpec("テスト", DummyEstimator, {"C": [0.1, 1.0]})

        grid = prefixed_param_grid(spec)

        assert grid == {f"{CLASSIFIER_STEP}__C": [0.1, 1.0]}

    def test_探索して最良の設定を返す(self) -> None:
        """小さなデータで探索を1件だけ実行し、結果の形を確かめる。

        成績の良し悪しは見ない。見ると、乱数や scikit-learn の版の違いで
        落ちるテストになる。
        """
        rng = np.random.default_rng(seed=0)
        features = rng.normal(size=(40, 3))
        target = (features[:, 0] > 0).astype(int)
        spec = ModelSpec(
            name="テスト",
            estimator_factory=_logistic_regression,
            param_grid={"C": [0.1, 1.0]},
        )

        result = train_with_grid_search(spec, features, target, cv_folds=3)

        assert result.name == "テスト"
        assert set(result.best_params) == {"C"}  # 接頭辞が外れている
        assert 0.0 <= result.cv_score <= 1.0
        assert hasattr(result.estimator, "predict")


def _logistic_regression():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression()


class Test訓練と検証の分割:
    def test_成功と失敗の比率を保つ(self) -> None:
        """偏ったデータをそのまま分割すると、検証側に失敗例が
        ほとんど入らないことがある。
        """
        features = np.arange(60).reshape(30, 2)
        target = np.array([1] * 20 + [0] * 10)

        _, x_test, _, y_test = split_train_test(features, target, test_size=0.2)

        assert len(x_test) == 6
        assert list(y_test).count(0) == 2  # 元の比率 1:2 が保たれる


class Test評価指標:
    def test_多数派を選ぶだけの正解率(self) -> None:
        """モデルの成績は、この値を上回って初めて意味を持つ。"""
        assert majority_class_accuracy([1] * 60 + [0] * 30) == pytest.approx(2 / 3)

    def test_空のデータでは基準を出せない(self) -> None:
        with pytest.raises(ValueError, match="空のデータ"):
            majority_class_accuracy([])

    def test_指標が手計算と一致する(self) -> None:
        """予測を固定した置き石で、4種類の指標を検算する。

        実際の内訳は 成功2件・失敗2件。予測は成功3件・失敗1件。
        - 当たった成功 2件、外した失敗 1件、当たった失敗 1件、見逃した成功 0件
        - 適合率 = 2/3、再現率 = 2/2 = 1.0
        """
        predictions = np.array([1, 1, 1, 0])
        actual = np.array([1, 1, 0, 0])

        scores = score_model("置き石", _FixedEstimator(predictions), None, actual)

        assert scores.accuracy == pytest.approx(0.75)
        assert scores.precision == pytest.approx(2 / 3)
        assert scores.recall == pytest.approx(1.0)
        assert scores.confusion == ConfusionCounts(
            true_negative=1, false_positive=1, false_negative=0, true_positive=2
        )

    def test_片側に寄った予測でも止まらない(self) -> None:
        """全部「失敗」と予測すると適合率が計算できなくなる。

        例外にせず 0 として残すのは、**片側に寄ったという事実が
        数字として見えるほうが原因を追いやすい** ため。
        """
        scores = score_model(
            "全部失敗",
            _FixedEstimator(np.zeros(4, dtype=int)),
            None,
            np.array([1, 1, 0, 0]),
        )

        assert scores.precision == 0.0
        assert scores.recall == 0.0
        assert scores.confusion.true_positive == 0


class _FixedEstimator:
    """常に決まった予測を返す置き石。指標の計算だけを検算するために使う。"""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, features: object) -> np.ndarray:
        return self._predictions


def test_比較表はF1の高い順に並ぶ() -> None:
    weak = score_model(
        "弱い", _FixedEstimator(np.array([0, 0, 0, 0])), None, np.array([1, 1, 0, 0])
    )
    strong = score_model(
        "強い", _FixedEstimator(np.array([1, 1, 0, 0])), None, np.array([1, 1, 0, 0])
    )

    table = comparison_table([weak, strong])

    assert list(table["モデル"]) == ["強い", "弱い"]
