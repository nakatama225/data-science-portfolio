"""成功率の集計。

「どの発射場が」「どの軌道向けが」「どの年が」成功しやすいかを数値で出す。
図にする前に数値で確かめられるようにしてあるのは、**目で見て納得した形が
実は集計の誤りだった**ということを避けるため。

集計の結果には件数を必ず添える。成功率 100% でも、それが 1 件のうち 1 件なのか
40 件のうち 40 件なのかで意味がまったく違う。件数の無い割合は読めない。
"""

from __future__ import annotations

from typing import Any

CLASS_COLUMN = "Class"
DATE_COLUMN = "Date"


def success_rate_by(
    frame: Any,
    group_column: str,
    target_column: str = CLASS_COLUMN,
) -> Any:
    """指定した列で分類したときの成功率と件数を返す。

    Args:
        frame: 目的変数を含む表。
        group_column: 分類に使う列（`LaunchSite`、`Orbit` など）。
        target_column: 成功を 1 とする列。

    Returns:
        `group_column` / `success_rate` / `launches` の3列を持つ表。
        成功率の高い順に並ぶ。成功率が同じ分類は、分類の値の順に並ぶ。

    Note:
        pandas の既定の並べ方は同点の順を保つと決まっておらず、17 行以上の表で
        入れ替える。同点の順を保つ並べ方（`kind="stable"`）を指定し、
        集計が並べた分類の値の順を残している。
    """
    grouped = frame.groupby(group_column)[target_column]
    summary = grouped.agg(success_rate="mean", launches="count").reset_index()
    return summary.sort_values(
        "success_rate", ascending=False, kind="stable"
    ).reset_index(drop=True)


def yearly_success_rate(
    frame: Any,
    date_column: str = DATE_COLUMN,
    target_column: str = CLASS_COLUMN,
) -> Any:
    """年ごとの成功率と件数を返す。

    Args:
        frame: 日付と目的変数を含む表。
        date_column: 日付の列。文字列でも日付型でも受け付ける。
        target_column: 成功を 1 とする列。

    Returns:
        `year` / `success_rate` / `launches` の3列を持つ表。年の昇順。

    Note:
        年の取り出しに文字列の先頭4文字を使わず、日付として解釈してから
        年を取り出している。日付の書き方は `2020-11-05` とは限らず、
        表記に依存した切り出しは書式が変わった瞬間に静かに壊れるため。
    """
    import pandas as pd

    with_year = frame.copy()
    with_year["year"] = pd.to_datetime(with_year[date_column]).dt.year

    grouped = with_year.groupby("year")[target_column]
    summary = grouped.agg(success_rate="mean", launches="count").reset_index()
    return summary.sort_values("year").reset_index(drop=True)


def payload_range_summary(
    frame: Any,
    bin_edges: tuple[float, ...] = (0.0, 3000.0, 6000.0, 10000.0, 20000.0),
    mass_column: str = "PayloadMass",
    target_column: str = CLASS_COLUMN,
) -> Any:
    """積荷の重量帯ごとの成功率と件数を返す。

    重量帯に区切るのは、1件ずつ点で見ても傾向が読めないため。
    区切り位置は既定で 3t / 6t / 10t としている。これは物理的な境界ではなく、
    件数が偏りすぎない位置を選んだもの。

    Args:
        frame: 重量と目的変数を含む表。
        bin_edges: 区切りの位置（キログラム）。昇順であること。
        mass_column: 重量の列。
        target_column: 成功を 1 とする列。

    Returns:
        `payload_range` / `success_rate` / `launches` の3列を持つ表。
        件数の合計は、渡した表の行数に等しい。

    Raises:
        ValueError: 区切りが昇順でない、または2つ未満のとき。重量が欠けた行か、
            区切りの外の重量があるとき。どちらもどの帯にも入らず、黙って件数から
            消えるので、止めて知らせる。
    """
    import pandas as pd

    if len(bin_edges) < 2:
        raise ValueError("区切りは2つ以上必要")
    if list(bin_edges) != sorted(bin_edges):
        raise ValueError(f"区切りが昇順でない: {bin_edges}")

    missing = int(frame[mass_column].isna().sum())
    if missing:
        raise ValueError(f"重量が欠けた行が {missing} 件ある。先に埋めるか除く")

    binned = frame.copy()
    binned["payload_range"] = pd.cut(
        binned[mass_column], bins=list(bin_edges), include_lowest=True
    )
    outside = int(binned["payload_range"].isna().sum())
    if outside:
        raise ValueError(f"区切りの外の重量が {outside} 件ある: 区切り {bin_edges}")

    grouped = binned.groupby("payload_range", observed=False)[target_column]
    summary = grouped.agg(success_rate="mean", launches="count").reset_index()
    return summary
