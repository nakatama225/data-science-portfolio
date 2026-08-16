"""SpaceX の Web API から打ち上げ記録を組み立てる。

## この API の構造上の都合

打ち上げ一覧は、ロケット・発射場・積荷・機体（コア）を **ID でしか持っていない**。

```
{"rocket": "5e9d0d95eda69973a809d1ec", "launchpad": "5e9e4501f509094ba4566f84", ...}
```

人が読める名前や、積荷の重量を得るには、ID ごとに別のエンドポイントを
引き直す必要がある。打ち上げ1件あたり4回、90件で360回の通信になる。

## 設計上の判断

**1. 同じ ID を二度引かない（`_lookup` のキャッシュ）**

ロケットの ID は数種類しかなく、同じ ID が何十回も出てくる。
素直に書くと同じ問い合わせを繰り返し、実行に数分かかるうえ
相手のサーバにも無駄な負荷をかける。一度引いた結果を覚えておくことで、
実際の通信回数は数百回から十数回に減る。

**2. 通信の方法を外から渡す（`JsonFetcher`）**

「URL を渡すと JSON が返るもの」という約束だけを決めて、実際の通信手段は
呼び出し側から渡す形にしている。こうしておくと、テストでは
あらかじめ用意した応答を返す偽物を渡せるので、**ネットワークに
つながっていなくてもデータ組み立ての正しさを検証できる**。

外部サービスは落ちるし、内容も変わる。テストがそれに巻き込まれると
「自分のコードが壊れたのか、相手が変わったのか」が区別できなくなる。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from spacex_capstone.config import (
    LAUNCH_DATE_CUTOFF,
    PAST_LAUNCHES_ENDPOINT,
    SPACEX_API_ROOT,
)

Json = Mapping[str, Any]


class JsonFetcher(Protocol):
    """URL を渡すと JSON を返すもの、という約束。

    実際の通信を行う `HttpJsonFetcher` と、テスト用にあらかじめ決めた応答を
    返す偽物の、両方がこの形を満たす。呼び出す側はどちらが渡されたかを
    知らないまま同じコードで動く。
    """

    def __call__(self, url: str) -> Any:  # pragma: no cover - 形の定義のみ
        ...


class HttpJsonFetcher:
    """`requests` を使って実際に通信する `JsonFetcher`。

    Args:
        timeout: 応答を待つ上限秒数。指定しないと相手が無応答のときに
            処理が永久に止まるため、必ず入れる。
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def __call__(self, url: str) -> Any:
        import requests  # 通信するときだけ読み込む（テストでは不要なため）

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True, slots=True)
class LaunchRecord:
    """打ち上げ1件分の記録。

    作ったあとは変更できない（`frozen=True`）。分析の途中で値が書き換わると、
    どの時点の値を見ているのか追えなくなるため、意図的に固定している。
    """

    flight_number: int
    """API 上の通し番号。"""

    date: dt.date
    booster_version: str
    """機体の型式（例: Falcon 9）。"""

    payload_mass_kg: float | None
    """積荷の重量。記録がない打ち上げでは `None` になる。"""

    orbit: str | None
    """投入先の軌道（例: LEO=地球低軌道、GTO=静止遷移軌道）。"""

    launch_site: str
    longitude: float
    latitude: float

    outcome: str
    """`<着陸成否> <着陸方式>` の2語。例: `True ASDS`（洋上の無人船に着陸成功）。"""

    flights: int
    """この機体が何回目の飛行か。"""

    grid_fins: bool
    """姿勢制御用の格子状の翼を装備しているか。"""

    reused: bool
    """再使用された機体か。"""

    legs: bool
    """着陸脚を装備しているか。"""

    landing_pad: str | None
    """着陸地点の ID。着陸を試みていない場合は `None`。"""

    block: float | None
    """機体の世代番号。大きいほど新しい設計。"""

    reused_count: int
    """この機体がそれまでに再使用された回数。"""

    serial: str | None
    """機体の個体識別番号（例: B1049）。"""


def filter_analysable_launches(
    launches: Iterable[Json],
    cutoff: dt.date = LAUNCH_DATE_CUTOFF,
) -> list[Json]:
    """分析の対象にできる打ち上げだけを残す。

    2つの条件で絞る。

    1. **機体も積荷も1つだけ** のもの。補助ブースタを2本足した大型機
       （Falcon Heavy）や、複数の積荷を相乗りさせた便は、
       「1回の打ち上げ＝1つの機体の着陸」という前提が崩れる。
    2. **`cutoff` 以前** の打ち上げ。もう1つのデータ源と期間を揃えるため。

    Args:
        launches: API が返した打ち上げ一覧。
        cutoff: この日までを対象にする。

    Returns:
        条件を満たす打ち上げのリスト。順序は入力のまま。
    """
    kept: list[Json] = []
    for launch in launches:
        cores = launch.get("cores") or []
        payloads = launch.get("payloads") or []
        if len(cores) != 1 or len(payloads) != 1:
            continue
        if _parse_utc_date(launch["date_utc"]) > cutoff:
            continue
        kept.append(launch)
    return kept


def _parse_utc_date(date_utc: str) -> dt.date:
    """API の日時文字列から日付だけを取り出す。

    形式は `2020-11-05T23:24:00.000Z`。末尾の `Z` は協定世界時を表すが、
    Python 3.11 より前の `fromisoformat` は `Z` を解釈できないため
    `+00:00` に置き換えてから渡す。
    """
    return dt.datetime.fromisoformat(date_utc.replace("Z", "+00:00")).date()


@dataclass
class SpaceXLaunchCollector:
    """打ち上げ一覧の ID を辿って、人が読める記録に組み立てる。

    Args:
        fetch: URL から JSON を取ってくるもの。
        api_root: 参照する API の入口。

    Attributes:
        request_count: 実際に外部へ問い合わせた回数。キャッシュが
            効いているかを確認するために数えている。
    """

    fetch: JsonFetcher
    api_root: str = SPACEX_API_ROOT
    request_count: int = field(default=0, init=False)
    _cache: dict[tuple[str, str], Json] = field(default_factory=dict, init=False)

    def collect(self, launches: Iterable[Json]) -> list[LaunchRecord]:
        """打ち上げ一覧をまとめて記録に変換する。"""
        return [self.build_record(launch) for launch in launches]

    def build_record(self, launch: Json) -> LaunchRecord:
        """打ち上げ1件を `LaunchRecord` に変換する。

        Args:
            launch: API が返した打ち上げ1件分。`cores` と `payloads` は
                それぞれ要素1つのリストである前提
                （`filter_analysable_launches` で保証する）。
        """
        rocket = self._lookup("rockets", launch["rocket"])
        launchpad = self._lookup("launchpads", launch["launchpad"])
        payload = self._lookup("payloads", launch["payloads"][0])

        core_entry = launch["cores"][0]
        core_id = core_entry.get("core")
        core = self._lookup("cores", core_id) if core_id else {}

        return LaunchRecord(
            flight_number=launch["flight_number"],
            date=_parse_utc_date(launch["date_utc"]),
            booster_version=rocket["name"],
            payload_mass_kg=payload.get("mass_kg"),
            orbit=payload.get("orbit"),
            launch_site=launchpad["name"],
            longitude=launchpad["longitude"],
            latitude=launchpad["latitude"],
            # 着陸の成否と方式は打ち上げ側にあり、機体側には無い。
            outcome=f"{core_entry.get('landing_success')} "
            f"{core_entry.get('landing_type')}",
            flights=core_entry["flight"],
            grid_fins=core_entry["gridfins"],
            reused=core_entry["reused"],
            legs=core_entry["legs"],
            landing_pad=core_entry.get("landpad"),
            block=core.get("block"),
            reused_count=core.get("reuse_count", 0),
            serial=core.get("serial"),
        )

    def _lookup(self, resource: str, identifier: str) -> Json:
        """ID から詳細を引く。一度引いた組み合わせは覚えておく。

        ロケットの型式は数種類しかなく、同じ ID が何十回も現れる。
        覚えておかないと、同じ答えを得るために同じ問い合わせを
        繰り返すことになる。
        """
        key = (resource, identifier)
        if key not in self._cache:
            self._cache[key] = self.fetch(f"{self.api_root}/{resource}/{identifier}")
            self.request_count += 1
        return self._cache[key]


def fetch_past_launches(fetch: JsonFetcher) -> Sequence[Json]:
    """過去の打ち上げ一覧を取得する。"""
    return fetch(PAST_LAUNCHES_ENDPOINT)


def records_to_frame(records: Iterable[LaunchRecord]) -> Any:
    """`LaunchRecord` の並びを表形式（DataFrame）に変換する。

    列名は、このあとの整形処理や過去のノートブックが使っている名前
    （`FlightNumber`、`PayloadMass` のような大文字始まり）に合わせている。
    Python の慣習とは違うが、**すでに保存されている CSV の列名と
    食い違わせないこと** を優先した。
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "FlightNumber": r.flight_number,
                "Date": r.date,
                "BoosterVersion": r.booster_version,
                "PayloadMass": r.payload_mass_kg,
                "Orbit": r.orbit,
                "LaunchSite": r.launch_site,
                "Outcome": r.outcome,
                "Flights": r.flights,
                "GridFins": r.grid_fins,
                "Reused": r.reused,
                "Legs": r.legs,
                "LandingPad": r.landing_pad,
                "Block": r.block,
                "ReusedCount": r.reused_count,
                "Serial": r.serial,
                "Longitude": r.longitude,
                "Latitude": r.latitude,
            }
            for r in records
        ]
    )
