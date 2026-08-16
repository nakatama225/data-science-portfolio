"""発射場の位置関係を地図に描く。

地図は `folium` で作る。出来上がるのは HTML なので、ノートブックに
そのまま表示することも、ファイルに保存してブラウザで開くこともできる。

## 地図で何を見るか

発射場の立地には制約がある。打ち上げが失敗したときに被害が及ばないよう
市街地から離れている必要がある一方、機材と人員を運ぶために鉄道や道路からは
遠すぎてはならない。地図に距離を書き込むと、その両立が数値で見える。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from spacex_capstone.analysis.geo import Coordinate, NamedPlace, haversine_distance_km

SUCCESS_COLOR = "green"
FAILURE_COLOR = "red"


def create_base_map(center: Coordinate, zoom: int = 5) -> Any:
    """指定した地点を中心にした地図を作る。

    Args:
        center: 地図の中心。
        zoom: 拡大率。数値が大きいほど拡大される。

    Returns:
        folium の Map。
    """
    import folium

    return folium.Map(location=[center.latitude, center.longitude], zoom_start=zoom)


def add_launch_sites(site_map: Any, sites: Iterable[NamedPlace]) -> Any:
    """発射場の位置に印と名前を置く。

    Args:
        site_map: 追加先の地図。
        sites: 発射場の名前と座標。

    Returns:
        印を追加した地図（引数と同じもの）。
    """
    import folium

    for site in sites:
        position = [site.coordinate.latitude, site.coordinate.longitude]
        site_map.add_child(
            folium.Circle(position, radius=1000, color="#d35400", fill=True).add_child(
                folium.Popup(site.name)
            )
        )
        site_map.add_child(
            folium.Marker(
                position,
                icon=folium.features.DivIcon(
                    icon_size=(120, 20),
                    icon_anchor=(0, 0),
                    html=(
                        '<div style="font-size:12px;color:#d35400">'
                        f"<b>{site.name}</b></div>"
                    ),
                ),
            )
        )
    return site_map


def add_launch_outcomes(site_map: Any, frame: Any) -> Any:
    """打ち上げ1件ごとの成否を、印の色で地図に重ねる。

    同じ場所に何十件も重なるため、近い印をまとめて表示する仕組み
    （クラスタ）を使う。まとめないと、印が重なって件数が読めない。

    Args:
        site_map: 追加先の地図。
        frame: `Lat` / `Long` / `class` を含む表。

    Returns:
        印を追加した地図。
    """
    import folium
    from folium.plugins import MarkerCluster

    cluster = MarkerCluster()
    site_map.add_child(cluster)

    for _, record in frame.iterrows():
        color = SUCCESS_COLOR if record["class"] == 1 else FAILURE_COLOR
        cluster.add_child(
            folium.Marker(
                [record["Lat"], record["Long"]],
                icon=folium.Icon(color=color),
            )
        )
    return site_map


def add_distance_line(
    site_map: Any,
    origin: Coordinate,
    destination: NamedPlace,
) -> float:
    """2点を線で結び、その距離を地図上に書き込む。

    Args:
        site_map: 追加先の地図。
        origin: 起点（発射場）。
        destination: 終点（海岸線・鉄道など）。

    Returns:
        2点間の距離（キロメートル）。
    """
    import folium

    distance = haversine_distance_km(origin, destination.coordinate)
    end = [destination.coordinate.latitude, destination.coordinate.longitude]

    site_map.add_child(
        folium.Marker(
            end,
            icon=folium.features.DivIcon(
                icon_size=(120, 20),
                icon_anchor=(0, 0),
                html=(
                    '<div style="font-size:12px;color:#d35400">'
                    f"<b>{destination.name} {distance:.2f} km</b></div>"
                ),
            ),
        )
    )
    site_map.add_child(
        folium.PolyLine(
            locations=[[origin.latitude, origin.longitude], end],
            weight=1,
        )
    )
    return distance
