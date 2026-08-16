"""テスト全体で使い回す道具立て。

## 外部サービスに触れない

このテストは Web API にも Wikipedia にも接続しない。代わりに、
本物と同じ形の応答を返す偽物（`FakeFetcher`）を渡す。

接続してしまうと、テストが落ちたときに原因が3つに増える
（自分のコードの誤り／相手のサービスの障害／相手のデータの変更）。
切り離しておけば、落ちた＝自分のコードが変わった、と読める。
"""

from __future__ import annotations

from typing import Any

import pytest

from spacex_capstone.config import SPACEX_API_ROOT


class FakeFetcher:
    """あらかじめ決めた応答だけを返す `JsonFetcher`。

    Attributes:
        calls: 呼ばれた URL の記録。同じ URL を二度引いていないかの
            確認に使う。
    """

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        if url not in self._responses:
            raise AssertionError(f"用意していない URL が呼ばれた: {url}")
        return self._responses[url]


@pytest.fixture
def api_responses() -> dict[str, Any]:
    """SpaceX API の応答を模したデータ。

    実際の応答から、このコードが読む項目だけを抜き出してある。
    """
    return {
        f"{SPACEX_API_ROOT}/rockets/rocket-f9": {"name": "Falcon 9"},
        f"{SPACEX_API_ROOT}/rockets/rocket-f1": {"name": "Falcon 1"},
        f"{SPACEX_API_ROOT}/launchpads/pad-ccafs": {
            "name": "CCAFS SLC 40",
            "longitude": -80.577366,
            "latitude": 28.561857,
        },
        f"{SPACEX_API_ROOT}/payloads/payload-1": {"mass_kg": 5000.0, "orbit": "LEO"},
        f"{SPACEX_API_ROOT}/payloads/payload-2": {"mass_kg": None, "orbit": "GTO"},
        f"{SPACEX_API_ROOT}/cores/core-b1049": {
            "block": 5.0,
            "reuse_count": 3,
            "serial": "B1049",
        },
    }


@pytest.fixture
def fetcher(api_responses: dict[str, Any]) -> FakeFetcher:
    return FakeFetcher(api_responses)


def make_launch(
    flight_number: int = 1,
    date_utc: str = "2020-01-07T02:19:00.000Z",
    payload: str = "payload-1",
    landing_success: bool | None = True,
    landing_type: str | None = "ASDS",
    core: str | None = "core-b1049",
) -> dict[str, Any]:
    """API の打ち上げ1件分に相当するデータを作る。

    既定値は「よくある1件」（Falcon 9・洋上着陸に成功）。
    テストごとに変えたい項目だけを引数で上書きする。
    """
    return {
        "flight_number": flight_number,
        "date_utc": date_utc,
        "rocket": "rocket-f9",
        "launchpad": "pad-ccafs",
        "payloads": [payload],
        "cores": [
            {
                "core": core,
                "flight": 4,
                "gridfins": True,
                "reused": True,
                "legs": True,
                "landpad": "pad-landing-1",
                "landing_success": landing_success,
                "landing_type": landing_type,
            }
        ],
    }


@pytest.fixture
def labelled_frame() -> Any:
    """目的変数付きの小さな打ち上げ表。

    成功2件・失敗1件で、発射場は2か所。集計の検算を手計算でできる大きさに
    してある。
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "FlightNumber": [1, 2, 3],
            "Date": ["2018-01-01", "2019-06-01", "2019-09-01"],
            "BoosterVersion": ["Falcon 9"] * 3,
            "PayloadMass": [3000.0, 6000.0, 9000.0],
            "Orbit": ["LEO", "GTO", "LEO"],
            "LaunchSite": ["CCAFS SLC 40", "KSC LC 39A", "CCAFS SLC 40"],
            "Outcome": ["True ASDS", "False ASDS", "True RTLS"],
            "Flights": [1, 2, 3],
            "GridFins": [True, True, False],
            "Reused": [False, True, True],
            "Legs": [True, True, False],
            "LandingPad": ["pad-1", "pad-1", None],
            "Block": [5.0, 5.0, 4.0],
            "ReusedCount": [0, 1, 2],
            "Serial": ["B1049", "B1051", "B1046"],
            "Class": [1, 0, 1],
        }
    )
