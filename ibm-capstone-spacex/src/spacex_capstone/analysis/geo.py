"""地図上の2点間の距離を求める。

発射場が海岸・鉄道・高速道路・市街地からどれだけ離れているかを測るために使う。
発射場の立地には理由があり、「打ち上げ失敗時に被害が及ばないだけの距離」と
「機材と人を運べるだけの近さ」の両立が求められる。

## なぜ単純な引き算では駄目なのか

緯度経度は地球の表面という曲面の上の座標なので、平面の座標のように
差の二乗和では距離にならない。とくに緯度が高い場所ほど、
経度1度あたりの実距離が短くなる。

ここでは球面上の2点間の距離を求める式（ハーサイン公式）を使う。
地球を完全な球とみなす近似だが、数十キロの範囲では誤差は 0.5% 未満で、
「海岸から何キロ離れているか」を語るには十分である。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

EARTH_RADIUS_KM: Final = 6373.0
"""距離の計算に使う地球の半径。

元の分析（`notebooks/06_launch-site-geography.ipynb` の `calculate_distance`）が
「地球のおおよその半径」として使っていた値を、そのまま使っている。
地球は完全な球ではなく赤道方向に膨らんでいるため、半径は緯度によって
6357〜6378km の幅がある。よく使われる平均の半径 6371km より 2km 大きく、
距離はおよそ 0.03% 長く出る。数十キロの距離なら 10m 前後の差で、
元の分析の数値を再現できるほうを優先して値を変えていない。
"""


@dataclass(frozen=True, slots=True)
class Coordinate:
    """地図上の1点。

    緯度と経度を別々の数値で持ち回すと、引数の順番を取り違えても
    型の上では気づけない。組にしておけば取り違えが起きない。
    """

    latitude: float
    """北緯を正とする角度（度）。"""

    longitude: float
    """東経を正とする角度（度）。"""

    def __post_init__(self) -> None:
        """値が地球上にありうる範囲に収まっているか確かめる。

        Raises:
            ValueError: 緯度が ±90 度、経度が ±180 度を超えるとき。
        """
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"緯度が範囲外: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"経度が範囲外: {self.longitude}")


def haversine_distance_km(origin: Coordinate, destination: Coordinate) -> float:
    """球面上の2点間の距離をキロメートルで返す。

    Args:
        origin: 起点。
        destination: 終点。

    Returns:
        2点を結ぶ地表面に沿った距離。同じ点なら 0。

    Note:
        起点と終点を入れ替えても結果は同じになる。
    """
    lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
    lat2, lon2 = math.radians(destination.latitude), math.radians(destination.longitude)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # 2点間の角度の差を、球面上の弧の長さに変換する。
    chord = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    # 式の上では 1 を超えないが、地球の反対側やその近くにある2点では、小数の計算の
    # 丸めで 1 をわずかに超えることがある（例：反対側どうしの (-87.5, -180) と
    # (87.5, 0)）。そのままだと 1 - chord が負になり、平方根が ValueError で止まるので、
    # 1 で頭打ちにする。
    chord = min(chord, 1.0)
    central_angle = 2 * math.atan2(math.sqrt(chord), math.sqrt(1 - chord))
    return EARTH_RADIUS_KM * central_angle


@dataclass(frozen=True, slots=True)
class NamedPlace:
    """名前の付いた地点。距離の一覧を人が読める形で出すために使う。"""

    name: str
    coordinate: Coordinate


def distances_from(
    origin: Coordinate,
    places: Iterable[NamedPlace],
) -> list[tuple[str, float]]:
    """起点から各地点までの距離を、近い順に並べて返す。

    Args:
        origin: 起点（発射場など）。
        places: 距離を測りたい地点の並び。

    Returns:
        `(地点名, 距離km)` の組を近い順に並べたリスト。
    """
    measured = [
        (place.name, haversine_distance_km(origin, place.coordinate))
        for place in places
    ]
    return sorted(measured, key=lambda item: item[1])
