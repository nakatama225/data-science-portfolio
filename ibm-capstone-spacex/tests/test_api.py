"""`collection.api` のテスト。

守りたいのは3つ。

1. ID を辿って作った記録が、元データの値と一致すること
2. 同じ ID を二度引かないこと（引くと実行時間が数分単位で伸びる）
3. 分析対象の絞り込みが、意図した条件で効いていること
"""

from __future__ import annotations

import datetime as dt

import pytest

from spacex_capstone.collection.api import (
    LaunchRecord,
    SpaceXLaunchCollector,
    filter_analysable_launches,
)
from spacex_capstone.config import SPACEX_API_ROOT

from .conftest import FakeFetcher, make_launch


def test_ID_を辿って打ち上げ記録を組み立てる(fetcher: FakeFetcher) -> None:
    collector = SpaceXLaunchCollector(fetch=fetcher)

    record = collector.build_record(make_launch())

    assert record == LaunchRecord(
        flight_number=1,
        date=dt.date(2020, 1, 7),
        booster_version="Falcon 9",
        payload_mass_kg=5000.0,
        orbit="LEO",
        launch_site="CCAFS SLC 40",
        longitude=-80.577366,
        latitude=28.561857,
        outcome="True ASDS",
        flights=4,
        grid_fins=True,
        reused=True,
        legs=True,
        landing_pad="pad-landing-1",
        block=5.0,
        reused_count=3,
        serial="B1049",
    )


def test_同じIDは二度引かない(fetcher: FakeFetcher) -> None:
    """3件とも同じロケット・発射場・機体を使う場合の通信回数を確かめる。

    素直に書くと 3件 × 4種類 = 12 回の通信になる。同じ ID を覚えておけば、
    実際に必要なのは種類の数だけ（ロケット・発射場・積荷・機体の4回）。
    """
    collector = SpaceXLaunchCollector(fetch=fetcher)

    collector.collect([make_launch(flight_number=n) for n in (1, 2, 3)])

    assert collector.request_count == 4
    assert len(fetcher.calls) == 4


def test_機体IDが無い打ち上げでも記録を作れる(fetcher: FakeFetcher) -> None:
    """初期の打ち上げには機体の個体情報が残っていない。

    その場合でも処理を止めず、機体に関する項目だけを空にして残す。
    落とすと、着陸を試みなかった時期の打ち上げが丸ごと分析から消える。
    """
    collector = SpaceXLaunchCollector(fetch=fetcher)

    record = collector.build_record(
        make_launch(core=None, landing_success=None, landing_type=None)
    )

    assert record.serial is None
    assert record.block is None
    assert record.reused_count is None
    assert record.outcome == "None None"


def test_未登録のURLを引こうとしたら気づける(fetcher: FakeFetcher) -> None:
    """テスト用の偽物が、用意していない URL を黙って許さないことの確認。

    許してしまうと、通信先を間違えたコードがテストを通過する。
    """
    collector = SpaceXLaunchCollector(fetch=fetcher)

    with pytest.raises(AssertionError, match="用意していない URL"):
        collector.build_record(make_launch() | {"rocket": "rocket-unknown"})


class Test絞り込み:
    """`filter_analysable_launches` の条件ごとの挙動。"""

    def test_機体が複数の打ち上げは除く(self) -> None:
        launch = make_launch()
        launch["cores"] = launch["cores"] * 2

        assert filter_analysable_launches([launch]) == []

    def test_積荷が複数の打ち上げは除く(self) -> None:
        launch = make_launch()
        launch["payloads"] = ["payload-1", "payload-2"]

        assert filter_analysable_launches([launch]) == []

    def test_期限より後の打ち上げは除く(self) -> None:
        after = make_launch(date_utc="2021-03-04T08:25:00.000Z")

        assert filter_analysable_launches([after]) == []

    def test_期限当日は含める(self) -> None:
        """境界の日を含めるか外すかは、二重に数えたり漏らしたりの元になる。

        `cutoff` は「この日まで」なので、当日は残す。
        """
        on_cutoff = make_launch(date_utc="2020-11-13T23:59:00.000Z")

        assert filter_analysable_launches([on_cutoff]) == [on_cutoff]

    def test_順序は入力のまま保たれる(self) -> None:
        launches = [make_launch(flight_number=n) for n in (3, 1, 2)]

        kept = filter_analysable_launches(launches)

        assert [launch["flight_number"] for launch in kept] == [3, 1, 2]


def test_通信先のURLが設定どおりに組み立てられる(fetcher: FakeFetcher) -> None:
    collector = SpaceXLaunchCollector(fetch=fetcher)

    collector.build_record(make_launch())

    assert f"{SPACEX_API_ROOT}/rockets/rocket-f9" in fetcher.calls
