"""予測したい答え（目的変数）を決める。

## 何を「成功」とみなすか

元データの結果欄は `True ASDS` のような2語の文字列である。

| 表記 | 意味 |
|---|---|
| `True ASDS` | 洋上の無人船に着陸して成功 |
| `False RTLS` | 発射場への帰還着陸を試みて失敗 |
| `None None` | 着陸を試みていない |

この分析の目的は「ロケットの1段目を回収できるか」の予測なので、
**着陸を試みていない打ち上げも「回収できていない」側に数える**。
試行のあった打ち上げだけを対象にすると、回収できた割合が実態より高く出る。

## 2つの定義がずれないようにする

失敗にあたる結果コードは `config.UNSUCCESSFUL_LANDING_OUTCOMES` にも
列挙してある。同じことが2か所に書かれている状態は、片方だけ直したときに
静かに食い違う。そのため `outcome_rule_matches_known_codes()` を用意し、
テストで両者の一致を確認している。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from spacex_capstone.config import UNSUCCESSFUL_LANDING_OUTCOMES

LANDING_CLASS_COLUMN = "Class"
"""目的変数の列名。1 が着陸成功、0 が失敗。"""


class LandingMethod(StrEnum):
    """着陸のやり方。

    値は元データの表記に合わせている。表記をそのまま値にしているのは、
    データを読むときと書くときで変換表を持たずに済むため。
    """

    DRONE_SHIP = "ASDS"
    """洋上に浮かべた無人船に降ろす。発射場に戻る燃料が足りないときに使う。"""

    GROUND_PAD = "RTLS"
    """発射場の近くの陸地に戻して降ろす。"""

    OCEAN = "Ocean"
    """海面に軟着水させる。機体は回収しない。"""

    NOT_ATTEMPTED = "None"
    """着陸を試みていない。"""

    @classmethod
    def from_raw(cls, token: str) -> LandingMethod:
        """元データの表記から対応する方式を返す。

        見たことのない表記が来た場合は `NOT_ATTEMPTED` に倒さず、
        例外を送出する。**知らない値を静かに既定値へ丸めると、
        データ側の変化に気づけなくなる** ため。

        Raises:
            ValueError: 対応する方式が無いとき。
        """
        try:
            return cls(token)
        except ValueError as exc:
            raise ValueError(f"未知の着陸方式: {token!r}") from exc


@dataclass(frozen=True, slots=True)
class LandingOutcome:
    """着陸の結果を、成否と方式に分けて持つ。

    文字列のまま扱うと `outcome.startswith("True")` のような判定が
    コードのあちこちに散らばる。一度この形に直しておけば、
    判定の書き方は1か所に閉じる。
    """

    succeeded: bool
    """着陸に成功したか。"""

    attempted: bool
    """着陸を試みたか。試みていない場合、`succeeded` は必ず False。"""

    method: LandingMethod

    @classmethod
    def parse(cls, raw: str) -> LandingOutcome:
        """`True ASDS` のような表記を読み解く。

        Args:
            raw: `<成否> <方式>` の2語。

        Raises:
            ValueError: 2語に分かれていない、または成否が
                `True` / `False` / `None` のいずれでもないとき。
        """
        parts = raw.split()
        if len(parts) != 2:
            raise ValueError(f"結果は2語である必要がある: {raw!r}")

        success_token, method_token = parts
        if success_token not in {"True", "False", "None"}:
            raise ValueError(f"未知の成否表記: {success_token!r}")

        method = LandingMethod.from_raw(method_token)
        return cls(
            succeeded=success_token == "True",
            attempted=method is not LandingMethod.NOT_ATTEMPTED,
            method=method,
        )


def is_successful_landing(raw: str) -> bool:
    """結果の表記から、着陸に成功したかどうかを判定する。"""
    return LandingOutcome.parse(raw).succeeded


def outcome_rule_matches_known_codes() -> bool:
    """設定に列挙された失敗コードが、判定規則と食い違っていないか確かめる。

    `config.UNSUCCESSFUL_LANDING_OUTCOMES` は、このデータに実際に現れた
    失敗コードの一覧である。判定規則を書き換えたとき、
    その一覧のどれかが「成功」に変わってしまっていないかをここで検出する。
    """
    return all(
        not LandingOutcome.parse(code).succeeded
        for code in UNSUCCESSFUL_LANDING_OUTCOMES
    )


def add_landing_class(frame: Any, outcome_column: str = "Outcome") -> Any:
    """表に目的変数の列を足した新しい表を返す。

    元の表は書き換えない。書き換えると、ノートブックでセルを
    実行し直したときに二重に処理がかかることがある。

    Args:
        frame: `outcome_column` を含む表。
        outcome_column: 結果の表記が入っている列名。

    Returns:
        `Class` 列（1=成功 / 0=失敗）を追加した表。
    """
    labelled = frame.copy()
    labelled[LANDING_CLASS_COLUMN] = (
        frame[outcome_column].map(is_successful_landing).astype(int)
    )
    return labelled
