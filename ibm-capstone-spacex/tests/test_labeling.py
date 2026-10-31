"""`preprocessing.labeling` のテスト。

目的変数の定義は、分析全体の土台になる。ここが変わると
すべての成功率・すべてのモデルの成績が変わるため、
**定義を変えたら必ずテストが落ちる** ようにしてある。
"""

from __future__ import annotations

import pytest

from spacex_capstone.config import UNSUCCESSFUL_LANDING_OUTCOMES
from spacex_capstone.preprocessing.labeling import (
    LandingMethod,
    LandingOutcome,
    add_landing_class,
    is_successful_landing,
    outcome_rule_matches_known_codes,
)


class Test結果の読み解き:
    @pytest.mark.parametrize(
        ("raw", "succeeded", "attempted", "method"),
        [
            ("True ASDS", True, True, LandingMethod.DRONE_SHIP),
            ("True RTLS", True, True, LandingMethod.GROUND_PAD),
            ("True Ocean", True, True, LandingMethod.OCEAN),
            ("False ASDS", False, True, LandingMethod.DRONE_SHIP),
            ("None ASDS", False, True, LandingMethod.DRONE_SHIP),
            ("None None", False, False, LandingMethod.NOT_ATTEMPTED),
        ],
    )
    def test_成否と方式に分解できる(
        self, raw: str, succeeded: bool, attempted: bool, method: LandingMethod
    ) -> None:
        outcome = LandingOutcome.parse(raw)

        assert outcome.succeeded is succeeded
        assert outcome.attempted is attempted
        assert outcome.method is method

    def test_着陸を試みていない打ち上げは失敗に数える(self) -> None:
        """この分析の目的は「回収できるか」なので、試みていない＝回収できていない。

        試行のあった打ち上げだけを対象にすると、成功率が実態より高く出る。
        """
        assert is_successful_landing("None None") is False


class Test不正な入力:
    """想定外の表記は、静かに片方へ倒さず例外にする。

    未知の値を既定値へ丸めると、データ側の変化に気づけないまま
    集計結果だけが変わる。
    """

    @pytest.mark.parametrize("raw", ["True", "", "True ASDS extra"])
    def test_2語でなければ例外(self, raw: str) -> None:
        with pytest.raises(ValueError, match="2語"):
            LandingOutcome.parse(raw)

    def test_未知の成否表記は例外(self) -> None:
        with pytest.raises(ValueError, match="未知の成否表記"):
            LandingOutcome.parse("Maybe ASDS")

    def test_未知の着陸方式は例外(self) -> None:
        with pytest.raises(ValueError, match="未知の着陸方式"):
            LandingOutcome.parse("True Balloon")

    @pytest.mark.parametrize("raw", ["True None", "False None"])
    def test_着陸を試みていないのに成否がある表記は例外(self, raw: str) -> None:
        """方式が `None`（試みていない）なら、成否も `None` のはず。

        `True None` を通すと、試みていないのに成功した打ち上げになり、
        「試みていなければ成功ではない」という約束が崩れる。
        どちらもこのデータには現れない。
        """
        with pytest.raises(ValueError, match="試みていない"):
            LandingOutcome.parse(raw)


def test_設定に列挙した失敗コードと判定規則が一致する() -> None:
    """同じことが2か所に書かれている状態への番人。

    `config.UNSUCCESSFUL_LANDING_OUTCOMES` は、このデータに現れた失敗コードの
    一覧。判定規則を書き換えたとき、そのどれかが「成功」に変わっていないかを
    検出する。
    """
    assert outcome_rule_matches_known_codes()
    assert len(UNSUCCESSFUL_LANDING_OUTCOMES) == 5


def test_表に目的変数の列を足す(labelled_frame) -> None:
    source = labelled_frame.drop(columns=["Class"])

    result = add_landing_class(source)

    assert list(result["Class"]) == [1, 0, 1]


def test_元の表を書き換えない(labelled_frame) -> None:
    """ノートブックでセルを実行し直したときに、二重に処理がかからないことの確認。"""
    source = labelled_frame.drop(columns=["Class"])

    add_landing_class(source)

    assert "Class" not in source.columns
