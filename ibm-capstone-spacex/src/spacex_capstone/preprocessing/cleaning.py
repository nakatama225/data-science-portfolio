"""分析対象を絞り、欠けている値を埋める。

ここで行う3つの処理には、それぞれ別の理由がある。

| 処理 | 理由 |
|---|---|
| Falcon 1 を除く | 設計も運用も別物で、着陸の成否を比べる相手にならない |
| 便番号を振り直す | 除外後に番号が飛ぶと「何回目の打ち上げか」がずれる |
| 積荷重量の欠損を埋める | 予測モデルは欠けた値をそのまま扱えない |

理由が別なので関数も分けている。「Falcon 1 の扱いを変えたい」と
「欠損の埋め方を変えたい」は別々に起きるため、同じ関数に入れると
片方を直すときにもう片方を読む羽目になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spacex_capstone.config import EXCLUDED_BOOSTER

FLIGHT_NUMBER_COLUMN = "FlightNumber"
PAYLOAD_MASS_COLUMN = "PayloadMass"


@dataclass(frozen=True, slots=True)
class ImputationReport:
    """欠損を埋めた結果の記録。

    埋めた事実と埋めた値を残すのは、あとから
    「この列の平均値が妙に揃っているのはなぜか」を追えるようにするため。
    埋めた記録が無いと、元から同じ値だったのか、埋めた結果なのか区別できない。
    """

    column: str
    filled_count: int
    """埋めた件数。"""

    fill_value: float
    """埋めるのに使った値。"""


def drop_excluded_boosters(
    frame: Any,
    booster_column: str = "BoosterVersion",
    excluded: str = EXCLUDED_BOOSTER,
) -> Any:
    """分析対象外の機体を落とした新しい表を返す。"""
    return frame[frame[booster_column] != excluded].copy()


def renumber_flights(frame: Any) -> Any:
    """便番号を 1 から振り直した新しい表を返す。

    対象外の機体を除くと、元の番号は歯抜けになる。この列は
    「打ち上げを重ねるほど成功しやすくなるか」を見るための
    経験の指標として使うので、連番でないと意味がずれる。
    """
    renumbered = frame.copy()
    renumbered[FLIGHT_NUMBER_COLUMN] = range(1, len(frame) + 1)
    return renumbered


def impute_payload_mass(
    frame: Any,
    column: str = PAYLOAD_MASS_COLUMN,
) -> tuple[Any, ImputationReport]:
    """積荷重量の欠損を、その列の平均値で埋める。

    平均値を選んだ理由は、**欠損の理由が「積荷が無い」ではなく
    「記録されていない」だから**。0 で埋めると、実際には
    重い積荷を積んでいた便が「空荷」として扱われ、
    重量と成否の関係が実態から離れる。

    Returns:
        埋めたあとの表と、何件をどの値で埋めたかの記録。
    """
    filled = frame.copy()
    missing_count = int(filled[column].isna().sum())
    fill_value = float(filled[column].mean())
    filled[column] = filled[column].fillna(fill_value)
    return filled, ImputationReport(
        column=column, filled_count=missing_count, fill_value=fill_value
    )


def prepare_launch_frame(frame: Any) -> tuple[Any, ImputationReport]:
    """3つの整形処理をこの順で通す。

    順番には依存関係がある。番号の振り直しは対象を絞ったあとでなければ
    意味がなく、平均値による欠損補完も対象を絞ったあとの平均でなければ
    分析対象外のデータが混ざる。

    Returns:
        整形後の表と、欠損補完の記録。
    """
    narrowed = drop_excluded_boosters(frame)
    renumbered = renumber_flights(narrowed)
    return impute_payload_mass(renumbered)
