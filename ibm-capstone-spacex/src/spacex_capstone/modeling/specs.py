"""予測モデルの「試す値の範囲」を定義する。

## 探索の定義と実行を分ける理由

予測モデルには、学習の前に人が決める設定値がある（設定値の例：
判断の厳しさ、木の深さ、参照する近傍の数）。どれが良いかは事前には
分からないので、候補を並べて総当たりで試し、最も成績の良い組み合わせを選ぶ。

このモジュールは **候補の並び方だけ** を持ち、学習の実行には関わらない。
そのため scikit-learn を読み込まない。分けたことで得られるものは3つ。

1. 「試す組み合わせが何通りになるか」を、1件も学習させずに数えられる
2. 値の指定ミス（空のリストを渡した等）を、学習を始める前に弾ける
3. 探索範囲の検査を、機械学習ライブラリの入っていない環境でも実行できる
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """1つの予測モデルについて、試す設定値の範囲をまとめたもの。

    Attributes:
        name: 結果の一覧に出す名前。
        estimator_factory: 呼ぶと未学習のモデルを新しく作る関数。
            出来上がったモデルではなく作る関数を持つのは、
            **同じモデルを使い回すと前の学習結果が残る** ため。
        param_grid: 設定値の名前と、試す候補の並び。
    """

    name: str
    estimator_factory: Callable[[], Any]
    param_grid: Mapping[str, Sequence[Any]]

    def __post_init__(self) -> None:
        """定義として成立しているかを確かめる。

        Raises:
            ValueError: 名前が空、候補が1つも無い設定値がある、
                または設定値が1つも無いとき。
        """
        if not self.name.strip():
            raise ValueError("モデル名が空")
        if not self.param_grid:
            raise ValueError(f"{self.name}: 試す設定値が1つも無い")
        for key, values in self.param_grid.items():
            if len(values) == 0:
                raise ValueError(f"{self.name}: 設定値 {key!r} の候補が空")

    @property
    def combination_count(self) -> int:
        """試す組み合わせの総数。

        各設定値の候補数を掛け合わせた数になる。候補を1つ増やすと
        総数が掛け算で増えるため、範囲を広げる前にこの数を見る。
        """
        return math.prod(len(values) for values in self.param_grid.values())


def total_fit_count(specs: Sequence[ModelSpec], cv_folds: int) -> int:
    """全モデルの学習回数を見積もる。

    総当たりの探索では、1つの組み合わせにつき `cv_folds` 回の学習を行う
    （データを分割して、分割ごとに学習と検証を繰り返すため）。
    実行前にこの数を見ておくと、想定外に時間がかかる探索範囲に気づける。

    Args:
        specs: 対象のモデル定義。
        cv_folds: データの分割数。

    Returns:
        学習が実行される回数の合計。

    Raises:
        ValueError: 分割数が 2 未満のとき。1 では検証用のデータが残らない。
    """
    if cv_folds < 2:
        raise ValueError(f"分割数は2以上である必要がある: {cv_folds}")
    return sum(spec.combination_count for spec in specs) * cv_folds
