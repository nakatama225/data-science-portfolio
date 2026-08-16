"""リポジトリに保存してあるデータそのものを検査する。

他のテストは小さな作り物のデータで論理を確かめている。ここだけは
**実際に保存されている 90 件のデータ** を読み、整形処理を通した結果が
保存済みの内容と一致することを確かめる。

この検査があると、整形処理を書き換えたときに
「昔の集計結果と食い違う変更をした」ことにその場で気づける。
数値が変わること自体は悪くないが、気づかずに変わるのは事故になる。
"""

from __future__ import annotations

import pandas as pd
import pytest

from spacex_capstone.config import PROCESSED_DATA_DIR
from spacex_capstone.preprocessing.features import build_feature_matrix
from spacex_capstone.preprocessing.labeling import add_landing_class

LABELLED_CSV = PROCESSED_DATA_DIR / "dataset_part_2.csv"
FEATURES_CSV = PROCESSED_DATA_DIR / "dataset_part_3.csv"

pytestmark = pytest.mark.skipif(
    not LABELLED_CSV.exists(),
    reason="保存済みデータが見つからない（データを含めずに取得した場合）",
)


@pytest.fixture(scope="module")
def stored_frame() -> pd.DataFrame:
    return pd.read_csv(LABELLED_CSV)


def test_保存されているのは90件(stored_frame: pd.DataFrame) -> None:
    assert len(stored_frame) == 90


def test_判定規則が保存済みの答えを再現する(stored_frame: pd.DataFrame) -> None:
    """`Outcome` 列から作り直した答えが、保存済みの `Class` 列と一致する。

    この分析の土台なので、1件でも食い違えば止める。
    """
    recomputed = add_landing_class(stored_frame.drop(columns=["Class"]))

    pd.testing.assert_series_equal(
        recomputed["Class"], stored_frame["Class"], check_dtype=False
    )


def test_成功60件_失敗30件(stored_frame: pd.DataFrame) -> None:
    """内訳を数値で固定しておく。

    「多いほうを選ぶだけで正解率 66.7%」という評価の基準は、
    この内訳から出てくる。内訳が変われば基準も変わる。
    """
    counts = stored_frame["Class"].value_counts()

    assert counts[1] == 60
    assert counts[0] == 30


def test_欠損が残っていない(stored_frame: pd.DataFrame) -> None:
    """整形済みデータに欠損があると、学習時に例外で止まる。

    `LandingPad` は「着陸を試みていない＝着陸地点が無い」という
    意味のある欠損なので、対象から外す。
    """
    checked = stored_frame.drop(columns=["LandingPad"])

    assert checked.isna().sum().sum() == 0


def test_説明変数の作り方が保存済みの表と一致する(stored_frame: pd.DataFrame) -> None:
    """`build_feature_matrix` が作る列が、保存済みの説明変数と同じであること。

    列の並びは問わない。ダミー変数化の順序は pandas の版で変わりうるが、
    **どの列があるか** が変われば意味が変わる。
    """
    stored_features = pd.read_csv(FEATURES_CSV)

    rebuilt = build_feature_matrix(stored_frame)

    assert set(rebuilt.columns) == set(stored_features.columns)
    assert len(rebuilt) == len(stored_features)
