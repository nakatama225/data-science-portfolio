"""打ち上げデータの図を描く。

図はすべて `matplotlib` の Figure として返す。表示も保存もしない。

## 図の描き方の方針

- 成功と失敗は **色だけでなく形でも区別する**。色覚の違いや白黒印刷でも
  読めるようにするため
- 軸のラベルには単位を書く。`PayloadMass` だけでは kg なのか t なのか分からない
- 割合を示す図には件数を添える。母数の分からない割合は判断に使えない
"""

from __future__ import annotations

from typing import Any

SUCCESS_STYLE = {"marker": "o", "label": "着陸成功"}
FAILURE_STYLE = {"marker": "x", "label": "着陸失敗"}


def plot_flight_number_vs_payload(frame: Any) -> Any:
    """便番号と積荷重量の散布図を、着陸の成否で描き分ける。

    横軸を便番号（＝運用の経験量）にしているのは、
    経験を積むにつれて重い積荷でも回収できるようになったか、
    という時間変化を1枚で見るため。

    Args:
        frame: `FlightNumber` / `PayloadMass` / `Class` を含む表。

    Returns:
        描画済みの Figure。
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(12, 5))
    for class_value, style in ((1, SUCCESS_STYLE), (0, FAILURE_STYLE)):
        subset = frame[frame["Class"] == class_value]
        axes.scatter(
            subset["FlightNumber"], subset["PayloadMass"], alpha=0.7, s=60, **style
        )

    axes.set_xlabel("便番号（打ち上げの通し番号）")
    axes.set_ylabel("積荷重量 (kg)")
    axes.set_title("便番号と積荷重量の関係")
    axes.legend()
    axes.grid(alpha=0.3)
    return figure


def plot_success_rate_by_category(summary: Any, category_column: str) -> Any:
    """分類ごとの成功率を棒グラフにする。

    各棒の上に件数を書き添える。成功率 100% が 1 件のうち 1 件なのか
    40 件のうち 40 件なのかで、読み方がまったく変わるため。

    Args:
        summary: `analysis.summaries.success_rate_by` が返した表。
        category_column: 分類の列名。

    Returns:
        描画済みの Figure。
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 5))
    bars = axes.bar(summary[category_column], summary["success_rate"])

    for bar, launches in zip(bars, summary["launches"], strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={launches}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes.set_ylim(0, 1.1)
    axes.set_xlabel(category_column)
    axes.set_ylabel("着陸成功率")
    axes.set_title(f"{category_column} ごとの着陸成功率")
    axes.grid(axis="y", alpha=0.3)
    return figure


def plot_yearly_trend(summary: Any) -> Any:
    """年ごとの成功率の推移を折れ線で描く。

    Args:
        summary: `analysis.summaries.yearly_success_rate` が返した表。

    Returns:
        描画済みの Figure。
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(summary["year"], summary["success_rate"], marker="o")

    axes.set_ylim(0, 1.05)
    axes.set_xlabel("年")
    axes.set_ylabel("着陸成功率")
    axes.set_title("着陸成功率の推移")
    axes.grid(alpha=0.3)
    return figure


def plot_confusion(counts: Any, title: str) -> Any:
    """予測と実際の内訳を、数値入りの表形式の図にする。

    Args:
        counts: `modeling.evaluation.ConfusionCounts`。
        title: 図の見出し（モデル名など）。

    Returns:
        描画済みの Figure。
    """
    import matplotlib.pyplot as plt

    grid = [
        [counts.true_negative, counts.false_positive],
        [counts.false_negative, counts.true_positive],
    ]

    figure, axes = plt.subplots(figsize=(5, 4))
    axes.imshow(grid, cmap="Blues")

    for row in range(2):
        for column in range(2):
            axes.text(column, row, str(grid[row][column]), ha="center", va="center")

    axes.set_xticks([0, 1], ["失敗と予測", "成功と予測"])
    axes.set_yticks([0, 1], ["実際は失敗", "実際は成功"])
    axes.set_title(title)
    return figure
