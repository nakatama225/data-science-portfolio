"""`analysis.geo` と `analysis.summaries` のテスト。

距離の計算は、答えが分かっている値と突き合わせる。
集計は、手で数えられる小さな表で検算する。
"""

from __future__ import annotations

import math

import pytest

from spacex_capstone.analysis.geo import (
    EARTH_RADIUS_KM,
    Coordinate,
    NamedPlace,
    distances_from,
    haversine_distance_km,
)
from spacex_capstone.analysis.summaries import (
    payload_range_summary,
    success_rate_by,
    yearly_success_rate,
)

# ケネディ宇宙センター第39A発射施設と、その東の海岸線。
LAUNCH_PAD_39A = Coordinate(latitude=28.573255, longitude=-80.646895)
NEAREST_COASTLINE = Coordinate(latitude=28.56405, longitude=-80.56807)


class Test座標:
    def test_範囲外の緯度を拒む(self) -> None:
        with pytest.raises(ValueError, match="緯度が範囲外"):
            Coordinate(latitude=91.0, longitude=0.0)

    def test_範囲外の経度を拒む(self) -> None:
        with pytest.raises(ValueError, match="経度が範囲外"):
            Coordinate(latitude=0.0, longitude=181.0)

    def test_境界の値は受け付ける(self) -> None:
        assert Coordinate(latitude=90.0, longitude=180.0).latitude == 90.0


class Test距離の計算:
    def test_同じ点なら0(self) -> None:
        assert haversine_distance_km(LAUNCH_PAD_39A, LAUNCH_PAD_39A) == 0.0

    def test_順番を入れ替えても同じ(self) -> None:
        forward = haversine_distance_km(LAUNCH_PAD_39A, NEAREST_COASTLINE)
        backward = haversine_distance_km(NEAREST_COASTLINE, LAUNCH_PAD_39A)

        assert forward == pytest.approx(backward)

    def test_発射場から海岸までの距離(self) -> None:
        """地図上で読み取れる約 7.7km と一致することを確かめる。"""
        distance = haversine_distance_km(LAUNCH_PAD_39A, NEAREST_COASTLINE)

        assert distance == pytest.approx(7.7, abs=0.2)

    def test_緯度1度は約111km(self) -> None:
        """よく知られた値との突き合わせ。式そのものの誤りを検出する。"""
        distance = haversine_distance_km(Coordinate(0.0, 0.0), Coordinate(1.0, 0.0))

        assert distance == pytest.approx(111.0, abs=1.0)

    def test_経度1度の距離は緯度によって変わる(self) -> None:
        """赤道付近では緯度1度とほぼ同じだが、高緯度では短くなる。

        平面として扱ってはいけないことの根拠になる性質。
        """
        at_equator = haversine_distance_km(Coordinate(0.0, 0.0), Coordinate(0.0, 1.0))
        at_60_north = haversine_distance_km(
            Coordinate(60.0, 0.0), Coordinate(60.0, 1.0)
        )

        assert at_60_north == pytest.approx(at_equator / 2, rel=0.01)

    def test_地球の反対側の2点でも止まらず半周を返す(self) -> None:
        """計算の途中の値が、小数の丸めで 1 をわずかに超える組。

        頭打ちにしないと、平方根に負の数を渡して ValueError で止まる。
        """
        distance = haversine_distance_km(
            Coordinate(-87.5, -180.0), Coordinate(87.5, 0.0)
        )

        assert distance == pytest.approx(math.pi * EARTH_RADIUS_KM)


def test_近い順に並べて返す() -> None:
    places = [
        NamedPlace("遠い街", Coordinate(28.60, -80.20)),
        NamedPlace("海岸線", NEAREST_COASTLINE),
    ]

    result = distances_from(LAUNCH_PAD_39A, places)

    assert [name for name, _ in result] == ["海岸線", "遠い街"]


class Test成功率の集計:
    def test_分類ごとの成功率と件数(self, labelled_frame) -> None:
        result = success_rate_by(labelled_frame, "LaunchSite")

        by_site = dict(zip(result["LaunchSite"], result["success_rate"], strict=True))
        counts = dict(zip(result["LaunchSite"], result["launches"], strict=True))

        assert by_site["CCAFS SLC 40"] == 1.0  # 2件とも成功
        assert by_site["KSC LC 39A"] == 0.0  # 1件が失敗
        assert counts["CCAFS SLC 40"] == 2

    def test_成功率の高い順に並ぶ(self, labelled_frame) -> None:
        result = success_rate_by(labelled_frame, "LaunchSite")

        assert list(result["success_rate"]) == [1.0, 0.0]

    def test_件数を必ず添える(self, labelled_frame) -> None:
        """割合だけでは、母数が1件なのか40件なのか読めない。"""
        result = success_rate_by(labelled_frame, "Orbit")

        assert "launches" in result.columns
        assert result["launches"].sum() == len(labelled_frame)


class Test年ごとの推移:
    def test_年で集約する(self, labelled_frame) -> None:
        result = yearly_success_rate(labelled_frame)

        assert list(result["year"]) == [2018, 2019]
        assert list(result["launches"]) == [1, 2]

    def test_年の昇順に並ぶ(self, labelled_frame) -> None:
        result = yearly_success_rate(labelled_frame)

        assert list(result["year"]) == sorted(result["year"])

    def test_日付の書き方が変わっても年を取れる(self, labelled_frame) -> None:
        """先頭4文字の切り出しではなく、日付として解釈していることの確認。"""
        other_format = labelled_frame.copy()
        other_format["Date"] = ["01/15/2018", "06/01/2019", "09/01/2019"]

        result = yearly_success_rate(other_format)

        assert list(result["year"]) == [2018, 2019]


class Test重量帯ごとの集計:
    def test_区切りごとに集計する(self, labelled_frame) -> None:
        """重量は 3000 / 6000 / 9000 kg なので、5000 で区切ると 1件 と 2件 に分かれる。"""
        result = payload_range_summary(labelled_frame, bin_edges=(0.0, 5000.0, 10000.0))

        assert list(result["launches"]) == [1, 2]

    def test_区切りが昇順でなければ止まる(self, labelled_frame) -> None:
        with pytest.raises(ValueError, match="昇順"):
            payload_range_summary(labelled_frame, bin_edges=(5000.0, 1000.0))

    def test_区切りが1つでは集計できない(self, labelled_frame) -> None:
        with pytest.raises(ValueError, match="2つ以上"):
            payload_range_summary(labelled_frame, bin_edges=(0.0,))
