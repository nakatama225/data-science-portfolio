"""README に載せている数値を、そのまま再現するための実行スクリプト。

README の「結果」節にある表（交差検証の平均正解率・検証データでの成績）は、
このスクリプトの出力をそのまま写したもの。数値の出どころを追えるように、
実行できる形で残している。

実行:
    uv run python ibm-capstone-spacex/scripts/run_pipeline.py

所要時間は数分。総当たり探索で 8,310 回の学習を行うため、
scikit-learn から収束に関する警告が多数出るが、探索範囲に
収束しない設定値の組み合わせが含まれているというだけで、異常ではない。
"""

from __future__ import annotations

import pandas as pd

from spacex_capstone.config import PROCESSED_DATA_DIR
from spacex_capstone.modeling.evaluation import (
    comparison_table,
    majority_class_accuracy,
    score_model,
)
from spacex_capstone.modeling.specs import total_fit_count
from spacex_capstone.modeling.training import (
    DEFAULT_CV_FOLDS,
    default_model_specs,
    split_train_test,
    train_all,
)
from spacex_capstone.preprocessing.features import (
    build_feature_matrix,
    split_features_and_target,
)


def main() -> None:
    frame = pd.read_csv(PROCESSED_DATA_DIR / "dataset_part_2.csv")
    matrix = build_feature_matrix(frame)
    features, target = split_features_and_target(matrix, frame)

    x_train, x_test, y_train, y_test = split_train_test(features, target)
    print(f"訓練 {len(x_train)} 件 / 検証 {len(x_test)} 件")
    print(f"説明変数 {features.shape[1]} 列")

    specs = default_model_specs()
    combinations = sum(spec.combination_count for spec in specs)
    print(f"探索する組み合わせ {combinations} 通り")
    print(f"学習の実行回数 {total_fit_count(specs, DEFAULT_CV_FOLDS)} 回")

    trained = train_all(specs, x_train, y_train)

    print("\n=== 交差検証（訓練データを10分割した平均正解率） ===")
    for model in trained:
        print(f"{model.name}: {model.cv_score:.4f}  設定={model.best_params}")

    scores = [score_model(m.name, m.estimator, x_test, y_test) for m in trained]
    print("\n=== 検証データでの成績 ===")
    print(comparison_table(scores).to_string(index=False))

    print(f"\n多数派を選ぶだけの正解率: {majority_class_accuracy(list(y_test)):.4f}")
    print(f"全データでの着陸成功率: {frame['Class'].mean():.4f}")


if __name__ == "__main__":
    main()
