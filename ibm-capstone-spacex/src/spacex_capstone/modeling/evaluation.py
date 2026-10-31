"""学習したモデルの出来を測る。

## 正解率だけを見ない理由

このデータは成功 60 件・失敗 30 件で、成功のほうが2倍多い。
**何も考えずに「全部成功」と答えるだけで正解率は 67% になる。**
つまり正解率 67% のモデルは、何も学習していないのと同じ成績である。

そのため、失敗を失敗と見抜けているかを別の指標で見る。

- **適合率（precision）** … 成功と予測したうちの的中率。
  「回収できる」と見込んだ機体のうち、実際に回収できた割合。
- **再現率（recall）** … 実際の成功のうち拾えた割合。
  回収できた機体を見落とさなかった割合。
- **F1** … 適合率と再現率の調和平均。両者のつり合い。

どちらを重く見るかは目的で変わる。回収前提で費用を見積もるなら、
「回収できると言ったのに落ちた」を減らしたいので適合率を重く見る。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """予測と実際の組み合わせ4通りの件数。

    「当たった／外れた」を1つの数にまとめる前の、生の内訳。
    正解率だけでは見えない偏り（失敗をすべて成功と予測している等）は、
    この4つの数を見ると分かる。
    """

    true_negative: int
    """失敗を失敗と予測できた件数。"""

    false_positive: int
    """失敗なのに成功と予測した件数。"""

    false_negative: int
    """成功なのに失敗と予測した件数。"""

    true_positive: int
    """成功を成功と予測できた件数。"""

    @property
    def total(self) -> int:
        return (
            self.true_negative
            + self.false_positive
            + self.false_negative
            + self.true_positive
        )


@dataclass(frozen=True, slots=True)
class ClassificationScores:
    """1つのモデルの成績。

    Attributes:
        name: モデル名。
        accuracy: 正解率。
        precision: 適合率。
        recall: 再現率。
        f1: 適合率と再現率の調和平均。
        confusion: 予測と実際の内訳。
    """

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: ConfusionCounts


def score_model(
    name: str,
    estimator: Any,
    features: Any,
    target: Any,
) -> ClassificationScores:
    """学習済みモデルを検証データで評価する。

    Args:
        name: 結果に付ける名前。
        estimator: 学習済みのモデル。
        features: 検証用の説明変数。
        target: 検証用の正解。

    Returns:
        各指標をまとめた成績。

    Note:
        適合率と再現率は、予測がすべて片側に寄ると計算できなくなる
        （0 での割り算になる）。その場合は 0 として扱う。
        エラーで止めるより、**「片側に寄った」という事実が
        数字として残るほう**が原因を追いやすい。
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    predicted = estimator.predict(features)
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return ClassificationScores(
        name=name,
        accuracy=float(accuracy_score(target, predicted)),
        precision=float(precision_score(target, predicted, zero_division=0)),
        recall=float(recall_score(target, predicted, zero_division=0)),
        f1=float(f1_score(target, predicted, zero_division=0)),
        confusion=ConfusionCounts(
            true_negative=int(true_negative),
            false_positive=int(false_positive),
            false_negative=int(false_negative),
            true_positive=int(true_positive),
        ),
    )


def majority_class_accuracy(target: Sequence[int]) -> float:
    """「多いほうの答えを常に選ぶ」だけで得られる正解率。

    モデルの正解率は、この値を上回って初めて意味を持つ。
    比較の基準を数値で置いておくと、成績を過大評価せずに済む。

    Raises:
        ValueError: データが空のとき。
    """
    if len(target) == 0:
        raise ValueError("空のデータでは基準を計算できない")
    positives = sum(1 for value in target if value == 1)
    return max(positives, len(target) - positives) / len(target)


def comparison_table(scores: Sequence[ClassificationScores]) -> Any:
    """複数モデルの成績を1つの表にまとめる。

    Returns:
        F1 の高い順に並べた表。F1 が同じモデルは、渡した順のまま並ぶ。

    Note:
        pandas の既定の並べ方は同点の順を保つと決まっておらず、入れ替えることがある
        （pandas 3.0.5・numpy 2.5.2 では、同点を含む 17 行以上の表で入れ替わった）。
        同点の順を保つ並べ方（`kind="stable"`）を指定している。
    """
    import pandas as pd

    table = pd.DataFrame(
        [
            {
                "モデル": score.name,
                "正解率": score.accuracy,
                "適合率": score.precision,
                "再現率": score.recall,
                "F1": score.f1,
                "見逃し": score.confusion.false_negative,
                "誤検知": score.confusion.false_positive,
            }
            for score in scores
        ]
    )
    return table.sort_values("F1", ascending=False, kind="stable").reset_index(
        drop=True
    )
