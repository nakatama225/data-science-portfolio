"""4種類の予測モデルを学習させ、設定値の総当たり探索を行う。

## 標準化を学習の中に閉じ込めている理由（重要）

説明変数は単位がばらばらである。積荷重量は数千（kg）、飛行回数は 1〜10 前後。
そのまま渡すと、数値の大きい列だけが結果を左右するモデルがある
（距離を測る種類のモデルが該当する）。そこで各列を平均 0・ばらつき 1 に
揃える処理（標準化）を挟む。

問題は **標準化に使う平均とばらつきを、どのデータから計算するか**。

```
避けるべき順番:  全データで平均を計算 → 訓練用と検証用に分割 → 学習
正しい順番    :  訓練用と検証用に分割 → 訓練用だけで平均を計算 → 学習
```

前者だと、検証用データの情報（平均とばらつき）が学習側に混ざる。
本番では未知のデータに対して予測するのに、検証時だけ答えの一部を
先に見ていることになり、**成績が実力より良く出る**。

これを避けるため、標準化とモデルを1つの流れ（Pipeline）にまとめ、
その流れごと分割・学習させている。こうすると分割のたびに、
その回の訓練データだけで平均が計算される。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spacex_capstone.modeling.specs import ModelSpec

CLASSIFIER_STEP = "classifier"
"""流れの中でモデルに付ける名前。設定値の指定に使う。"""

DEFAULT_CV_FOLDS = 10
"""データの分割数。

90 件しかないため、分割数を大きくすると1回あたりの検証データが
9 件程度になり、成績が偶然に左右されやすい。一方で小さくすると
学習に使えるデータが減る。元の分析に合わせて 10 としている。
"""

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 2
"""分割の乱数の種。

固定しているのは、実行するたびに違う分割になると成績が変わり、
モデル間の比較ができなくなるため。値そのものに意味は無い。
"""


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """1つのモデルについての探索結果。

    Attributes:
        name: モデル名。
        best_params: 最も成績の良かった設定値の組み合わせ。
        cv_score: 訓練データを分割して測った平均正解率。
        estimator: 最良の設定で学習し直したモデル本体。予測に使える。
    """

    name: str
    best_params: Mapping[str, Any]
    cv_score: float
    estimator: Any


def default_model_specs() -> tuple[ModelSpec, ...]:
    """この分析で比較する4つのモデルを返す。

    性質の異なるものを選んでいる。

    | モデル | 性質 |
    |---|---|
    | ロジスティック回帰 | 各要因の効き方を線形で表す。結果の説明がしやすい |
    | サポートベクターマシン | 境界線を引く位置を余裕をもって決める |
    | 決定木 | 条件分岐の連なりで判断する。判断の過程を図にできる |
    | k近傍法 | 似た過去の打ち上げを探して多数決する |

    Returns:
        モデル定義の並び。

    Note:
        乱数を使うモデルには `random_state` を固定して渡している。
        決定木は分割に使う特徴量を無作為に選ぶ設定（`max_features='sqrt'`、
        `splitter='random'`）を探索範囲に含むため、**種を固定しないと
        実行のたびに結果が変わる**。実際に固定前は、交差検証の平均正解率が
        0.905 と 0.916、検証データの正解率が 0.722 と 0.667 に振れた。

        結果が再現しないと、モデルを改良したのか偶然良く出たのかを
        区別できなくなる。
    """
    from functools import partial

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return (
        ModelSpec(
            name="ロジスティック回帰",
            # このモデルは今の設定（lbfgs）では乱数を使わないが、
            # 種を受け付けるものには一律で渡す。設定を変えたときに
            # 固定し忘れる余地を残さないため。
            estimator_factory=partial(
                LogisticRegression, random_state=DEFAULT_RANDOM_STATE
            ),
            # C は「訓練データへの当てはめをどれだけ優先するか」。
            # 小さいほど係数が抑えられ、過剰な当てはめを避ける。
            param_grid={"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]},
        ),
        ModelSpec(
            name="サポートベクターマシン",
            estimator_factory=partial(SVC, random_state=DEFAULT_RANDOM_STATE),
            param_grid={
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "C": np.logspace(-3, 3, 5).tolist(),
                "gamma": np.logspace(-3, 3, 5).tolist(),
            },
        ),
        ModelSpec(
            name="決定木",
            estimator_factory=partial(
                DecisionTreeClassifier, random_state=DEFAULT_RANDOM_STATE
            ),
            param_grid={
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
                "max_depth": [2 * n for n in range(1, 10)],
                "max_features": [None, "sqrt"],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5, 10],
            },
        ),
        ModelSpec(
            name="k近傍法",
            estimator_factory=KNeighborsClassifier,
            param_grid={
                "n_neighbors": list(range(1, 11)),
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],
            },
        ),
    )


def build_pipeline(spec: ModelSpec) -> Any:
    """標準化とモデルを1つの流れにまとめる。

    この流れ全体を分割・学習の対象にすることで、標準化の基準値が
    その回の訓練データだけから計算されるようになる。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (CLASSIFIER_STEP, spec.estimator_factory()),
        ]
    )


def prefixed_param_grid(spec: ModelSpec) -> dict[str, list[Any]]:
    """設定値の名前に、流れの中での位置を表す接頭辞を付ける。

    `C` という設定値は、流れの中では `classifier__C` と呼ぶ必要がある
    （標準化の側にも設定値があるため、どちらのものかを区別する）。

    この変換をここで行うことで、`ModelSpec` の側は流れの構造を
    知らずに済む。探索範囲の定義は「モデルの設定値」だけを語り、
    実行方法の都合はこの層が引き受ける。
    """
    return {
        f"{CLASSIFIER_STEP}__{key}": list(values)
        for key, values in spec.param_grid.items()
    }


def train_with_grid_search(
    spec: ModelSpec,
    features: Any,
    target: Any,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> TrainedModel:
    """1つのモデルについて総当たり探索を行い、最良の設定で学習する。

    Args:
        spec: 試す設定値の範囲。
        features: 訓練用の説明変数。
        target: 訓練用の目的変数。
        cv_folds: データの分割数。

    Returns:
        最良の設定と、その設定で学習したモデル。
    """
    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(
        estimator=build_pipeline(spec),
        param_grid=prefixed_param_grid(spec),
        cv=cv_folds,
    )
    search.fit(features, target)

    return TrainedModel(
        name=spec.name,
        # 接頭辞を外して、指定したときと同じ名前で結果を返す。
        best_params={
            key.removeprefix(f"{CLASSIFIER_STEP}__"): value
            for key, value in search.best_params_.items()
        },
        cv_score=float(search.best_score_),
        estimator=search.best_estimator_,
    )


def train_all(
    specs: Sequence[ModelSpec],
    features: Any,
    target: Any,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> list[TrainedModel]:
    """複数のモデルを同じ条件で学習させる。"""
    return [train_with_grid_search(spec, features, target, cv_folds) for spec in specs]


def split_train_test(
    features: Any,
    target: Any,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any, Any, Any, Any]:
    """データを訓練用と検証用に分ける。

    Args:
        features: 説明変数。
        target: 目的変数。
        test_size: 検証用に回す割合。
        random_state: 分割の乱数の種。

    Returns:
        `(訓練用説明変数, 検証用説明変数, 訓練用目的変数, 検証用目的変数)`。

    Note:
        目的変数の内訳（成功 60 件・失敗 30 件）に偏りがあるため、
        分割時に内訳の比率を保つよう指定している。指定しないと、
        検証用データに失敗例がほとんど入らない分割が起こりうる。
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
