"""`preprocessing.cleaning` と `preprocessing.features` のテスト。

整形処理の誤りは、実行時に例外を出さず **数値だけが静かにずれる** 形で現れる。
そのため「落ちないこと」ではなく「意図した値になっていること」を確かめる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacex_capstone.preprocessing.cleaning import (
    drop_excluded_boosters,
    impute_payload_mass,
    prepare_launch_frame,
    renumber_flights,
)
from spacex_capstone.preprocessing.features import (
    CATEGORICAL_FEATURES,
    build_feature_matrix,
    split_features_and_target,
)


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Falcon 1 が混ざり、積荷重量に欠損がある表。"""
    return pd.DataFrame(
        {
            "FlightNumber": [1, 2, 3, 4],
            "BoosterVersion": ["Falcon 1", "Falcon 9", "Falcon 9", "Falcon 9"],
            "PayloadMass": [20.0, 3000.0, np.nan, 5000.0],
        }
    )


class Test対象の絞り込み:
    def test_Falcon1を除く(self, mixed_frame: pd.DataFrame) -> None:
        result = drop_excluded_boosters(mixed_frame)

        assert len(result) == 3
        assert "Falcon 1" not in set(result["BoosterVersion"])

    def test_元の表を書き換えない(self, mixed_frame: pd.DataFrame) -> None:
        drop_excluded_boosters(mixed_frame)

        assert len(mixed_frame) == 4


class Test便番号の振り直し:
    def test_1から連番になる(self, mixed_frame: pd.DataFrame) -> None:
        narrowed = drop_excluded_boosters(mixed_frame)

        result = renumber_flights(narrowed)

        assert list(result["FlightNumber"]) == [1, 2, 3]

    def test_行の順序は変えない(self, mixed_frame: pd.DataFrame) -> None:
        """番号だけを振り直し、並べ替えはしない。

        並べ替えると、他の列との対応が崩れる。
        """
        narrowed = drop_excluded_boosters(mixed_frame)

        result = renumber_flights(narrowed)

        assert list(result["PayloadMass"].fillna(-1)) == [3000.0, -1, 5000.0]


class Test欠損の補完:
    def test_平均値で埋める(self, mixed_frame: pd.DataFrame) -> None:
        narrowed = drop_excluded_boosters(mixed_frame)

        filled, report = impute_payload_mass(narrowed)

        # Falcon 1 を除いた後の平均は (3000 + 5000) / 2 = 4000
        assert report.fill_value == 4000.0
        assert report.filled_count == 1
        assert filled["PayloadMass"].isna().sum() == 0

    def test_絞り込みの後で平均を取る(self, mixed_frame: pd.DataFrame) -> None:
        """絞り込む前に平均を取ると、分析対象外のデータが混ざる。

        Falcon 1 の 20kg を含めた平均は約 2673 で、正しい 4000 と大きく違う。
        """
        _, report = prepare_launch_frame(mixed_frame)

        assert report.fill_value == 4000.0

    def test_埋めた記録が残る(self, mixed_frame: pd.DataFrame) -> None:
        """あとから「この値は埋めたものか」を追えるようにする。"""
        _, report = prepare_launch_frame(mixed_frame)

        assert report.column == "PayloadMass"
        assert report.filled_count == 1


class Test説明変数の作成:
    def test_分類項目が0と1の列に開かれる(self, labelled_frame) -> None:
        matrix = build_feature_matrix(labelled_frame)

        # 元の表に LEO と GTO があるので、その2本に開かれる。
        assert "Orbit_LEO" in matrix.columns
        assert "Orbit_GTO" in matrix.columns
        assert list(matrix["Orbit_LEO"]) == [1.0, 0.0, 1.0]

    def test_元の分類列は残さない(self, labelled_frame) -> None:
        matrix = build_feature_matrix(labelled_frame)

        for column in CATEGORICAL_FEATURES:
            assert column not in matrix.columns

    def test_すべての列が同じ型になる(self, labelled_frame) -> None:
        """真偽値と数値が混ざったままだと、モデルによって扱いが変わる。"""
        matrix = build_feature_matrix(labelled_frame)

        assert set(matrix.dtypes.astype(str)) == {"float64"}

    def test_行数は変わらない(self, labelled_frame) -> None:
        matrix = build_feature_matrix(labelled_frame)

        assert len(matrix) == len(labelled_frame)

    def test_必要な列が足りなければ列名を示して止まる(self, labelled_frame) -> None:
        broken = labelled_frame.drop(columns=["Block"])

        with pytest.raises(KeyError, match="Block"):
            build_feature_matrix(broken)


class Test説明変数と目的変数の取り出し:
    def test_行数が揃っていれば取り出せる(self, labelled_frame) -> None:
        matrix = build_feature_matrix(labelled_frame)

        features, target = split_features_and_target(matrix, labelled_frame)

        assert len(features) == 3
        assert list(target) == [1, 0, 1]

    def test_行数が食い違えば止まる(self, labelled_frame) -> None:
        """別の打ち上げの答えを見て学習することを防ぐ。"""
        matrix = build_feature_matrix(labelled_frame)

        with pytest.raises(ValueError, match="行数が一致しない"):
            split_features_and_target(matrix, labelled_frame.iloc[:2])
