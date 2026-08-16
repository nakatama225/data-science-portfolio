"""文字列で書かれた項目を、予測モデルが扱える数値に変換する。

## 何をしているか

`Orbit` 列には `LEO` `GTO` `SSO` のような記号が入っている。予測モデルは
文字列を扱えないため、**「その軌道かどうか」を 0 と 1 で表す列に開く**。

```
Orbit                Orbit_LEO  Orbit_GTO  Orbit_SSO
LEO        ──▶            1          0          0
GTO                       0          1          0
```

この変換をダミー変数化（one-hot encoding）と呼ぶ。`LEO` を 1、`GTO` を 2 の
ように番号で表さないのは、番号にすると **軌道のあいだに大小関係が
あることになってしまう** ため。`GTO` が `LEO` の2倍という意味は無い。

## 既知の弱点

`Serial`（機体の個体識別番号）もダミー変数化している。機体は 60 種類以上
あるため、この1列だけで 60 本以上の列に増える。データが 90 件しかない中で
説明変数が数十本になると、**モデルが「特定の機体番号」を丸暗記して
高い正解率を出す** 危険がある。

元の分析の手順に合わせてこの形を保っているが、実運用に載せるなら
`Serial` は落とすか、機体の世代（`Block`）に集約するのが妥当である。
"""

from __future__ import annotations

from typing import Any, Final

NUMERIC_FEATURES: Final = (
    "FlightNumber",
    "PayloadMass",
    "Flights",
    "GridFins",
    "Reused",
    "Legs",
    "Block",
    "ReusedCount",
)
"""そのまま数値として使える列。真偽値（`GridFins` など）も 0/1 として扱える。"""

CATEGORICAL_FEATURES: Final = (
    "Orbit",
    "LaunchSite",
    "LandingPad",
    "Serial",
)
"""ダミー変数化する列。"""

FEATURE_COLUMNS: Final = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_feature_matrix(frame: Any) -> Any:
    """説明変数の表を作る。

    Args:
        frame: 整形済みの打ち上げ表。`FEATURE_COLUMNS` をすべて含むこと。

    Returns:
        すべての列が `float64` の表。

    Raises:
        KeyError: 必要な列が足りないとき。どの列が足りないかを示す。

    Note:
        最後にすべてを `float64` に揃えているのは、真偽値の列
        （`True` / `False`）と数値の列が混ざったままだと、
        予測モデルの実装によって扱いが変わるため。型を1つに揃えておけば、
        どのモデルに渡しても同じ意味になる。
    """
    import pandas as pd

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"説明変数に必要な列が足りない: {missing}")

    selected = frame[list(FEATURE_COLUMNS)]
    expanded = pd.get_dummies(selected, columns=list(CATEGORICAL_FEATURES))
    return expanded.astype("float64")


def split_features_and_target(
    features: Any,
    labelled_frame: Any,
    target_column: str = "Class",
) -> tuple[Any, Any]:
    """説明変数と目的変数を、行の並びを保ったまま取り出す。

    Args:
        features: `build_feature_matrix` が作った表。
        labelled_frame: 目的変数の列を持つ表。
        target_column: 目的変数の列名。

    Returns:
        `(説明変数, 目的変数)` の組。目的変数は 1 次元の配列。

    Raises:
        ValueError: 2つの表の行数が食い違うとき。**行数が違うまま
            学習させると、別の打ち上げの答えを見て学習することになる**。
            静かに間違うより、ここで止める。
    """
    if len(features) != len(labelled_frame):
        raise ValueError(
            f"行数が一致しない: 説明変数 {len(features)} 件 / "
            f"目的変数 {len(labelled_frame)} 件"
        )
    return features, labelled_frame[target_column].to_numpy()
