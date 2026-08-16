"""`collection.scraping` のテスト。

HTML の表は、人が読むための飾り（改行・リンク・脚注）を含む。
ここでは **飾りの入り方ごとに1つずつ** 検証する。飾りの入り方は
ページの編集で変わるため、まとめて1件のテストにすると
どの飾りで壊れたのかが分からない。
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from spacex_capstone.collection.scraping import (
    WikipediaLaunchTableParser,
    extract_column_name,
    parse_booster_version,
    parse_date_and_time,
    parse_landing_status,
    parse_payload_mass,
)


def cell(html: str):
    """HTML の断片を1つのセルとして読み込む。"""
    return BeautifulSoup(f"<table><tr>{html}</tr></table>", "html.parser").find(
        ["td", "th"]
    )


class Test積荷重量:
    """`parse_payload_mass` は表記のゆれを吸収する必要がある。"""

    @pytest.mark.parametrize(
        ("html", "expected"),
        [
            ("<td>500 kg</td>", 500.0),
            ("<td>15,600 kg</td>", 15600.0),  # 桁区切り
            ("<td>2,257 kg (4,976 lb)</td>", 2257.0),  # ポンド換算が続く
            ("<td>1234.5 kg</td>", 1234.5),  # 小数
            ("<td>500&nbsp;kg</td>", 500.0),  # 特殊な空白
        ],
    )
    def test_数値として取り出せる(self, html: str, expected: float) -> None:
        assert parse_payload_mass(cell(html)) == expected

    @pytest.mark.parametrize("html", ["<td></td>", "<td>Classified</td>", "<td>—</td>"])
    def test_記載が無ければNoneを返す(self, html: str) -> None:
        """0 ではなく None を返すことを固定する。

        0 を返すと「空荷だった」と区別できなくなり、平均値が実際より
        軽くなる。欠損は欠損として後段に渡す。
        """
        assert parse_payload_mass(cell(html)) is None


def test_日付と時刻を分けて取り出す() -> None:
    """日付・時刻・脚注が1つのセルに同居している場合。"""
    tag = cell("<td>4 June 2010,<br/>18:45<sup>[12]</sup></td>")

    assert parse_date_and_time(tag) == ("4 June 2010", "18:45")


def test_日付だけのセルでも落ちない() -> None:
    """時刻が無い行があっても、時刻を空文字にして処理を続ける。"""
    assert parse_date_and_time(cell("<td>4 June 2010</td>")) == ("4 June 2010", "")


class Test機体の型式:
    def test_型式と個体番号を繋いで返す(self) -> None:
        """保存済みの取得結果（`F9 v1.0B0003.1`）と同じ形になること。"""
        tag = cell("<td>F9 v1.0<sup>[1]</sup>B0003.1<sup>[2]</sup></td>")

        assert parse_booster_version(tag) == "F9 v1.0B0003.1"

    def test_末尾の改行があってもなくても同じ結果(self) -> None:
        """HTML の空白は表示に影響しないため、ページの編集で簡単に増減する。

        空白の有無で結果が変わる読み方をしていないことを、ここで固定する。
        """
        without_newline = cell("<td>F9 v1.1<sup>[1]</sup>B1003<sup>[2]</sup></td>")
        with_newline = cell("<td>F9 v1.1<sup>[1]</sup>B1003<sup>[2]</sup>\n</td>")

        assert parse_booster_version(without_newline) == "F9 v1.1B1003"
        assert parse_booster_version(with_newline) == "F9 v1.1B1003"

    def test_脚注が無い書き方にも対応する(self) -> None:
        tag = cell('<td><a href="/wiki/Falcon_9">F9 B5</a></td>')

        assert parse_booster_version(tag) == "F9 B5"

    def test_元のHTMLを書き換えない(self) -> None:
        """同じ表を別の目的でもう一度読めるようにする。

        脚注を取り除く処理が元の構造に及ぶと、2回目に読んだときには
        脚注が消えた表を読むことになる。
        """
        tag = cell("<td>F9 v1.0<sup>[1]</sup>B0003.1<sup>[2]</sup></td>")

        parse_booster_version(tag)

        assert tag.find("sup") is not None


def test_着陸結果は先頭の語だけを取る() -> None:
    tag = cell("<td>Success<br/>(drone ship)<sup>[3]</sup></td>")

    assert parse_landing_status(tag) == "Success"


class Test列名の取り出し:
    def test_飾りを取り除いた列名を返す(self) -> None:
        tag = cell("<th>Launch<br/>site<sup>[4]</sup></th>")

        assert extract_column_name(tag) == "Launch site"

    def test_数字だけの見出しはNone(self) -> None:
        """脚注番号だけの見出しは列名ではない。"""
        assert extract_column_name(cell("<th>12</th>")) is None

    def test_空の見出しはNone(self) -> None:
        assert extract_column_name(cell("<th></th>")) is None

    def test_元のHTMLを書き換えない(self) -> None:
        tag = cell("<th>Launch<br/>site<sup>[4]</sup></th>")

        extract_column_name(tag)

        assert tag.find("sup") is not None


LAUNCH_TABLE_HTML = """
<table class="wikitable plainrowheaders collapsible">
  <tr><th scope="col">Flight No.</th><td>Date</td></tr>
  <tr>
    <th scope="row">1</th>
    <td>4 June 2010,<br/>18:45</td>
    <td>F9 v1.0<sup>[1]</sup>B0003<sup>[2]</sup></td>
    <td><a href="#">CCAFS</a></td>
    <td><a href="#">Dragon Spacecraft Qualification Unit</a></td>
    <td></td>
    <td><a href="#">LEO</a></td>
    <td><a href="#">SpaceX</a></td>
    <td>Success</td>
    <td>Failure<sup>[9]</sup></td>
  </tr>
  <tr>
    <th scope="row">2</th>
    <td>8 December 2010,<br/>15:43</td>
    <td>F9 v1.0<sup>[1]</sup>B0004<sup>[2]</sup></td>
    <td><a href="#">CCAFS</a></td>
    <td><a href="#">Dragon demo flight C1</a></td>
    <td>525 kg</td>
    <td><a href="#">LEO</a></td>
    <td></td>
    <td>Success</td>
    <td>Failure</td>
  </tr>
  <tr><td colspan="9">注釈の行。打ち上げ記録ではない。</td></tr>
</table>
"""


class Test表全体の読み取り:
    @pytest.fixture
    def launches(self):
        soup = BeautifulSoup(LAUNCH_TABLE_HTML, "html.parser")
        return WikipediaLaunchTableParser().parse(soup)

    def test_打ち上げ行だけを拾う(self, launches) -> None:
        """見出し行と注釈行を除いた2件だけが残る。"""
        assert [launch.flight_no for launch in launches] == [1, 2]

    def test_各項目が正しい列から取れている(self, launches) -> None:
        first = launches[0]

        assert first.date == "4 June 2010"
        assert first.time == "18:45"
        assert first.launch_site == "CCAFS"
        assert first.orbit == "LEO"
        assert first.customer == "SpaceX"
        assert first.launch_outcome == "Success"
        assert first.booster_landing == "Failure"

    def test_重量の有無が行ごとに正しく反映される(self, launches) -> None:
        assert launches[0].payload_mass_kg is None
        assert launches[1].payload_mass_kg == 525.0

    def test_顧客のリンクが無い行はVariousになる(self, launches) -> None:
        """顧客が複数の場合、ページ上はリンクが張られていない。"""
        assert launches[1].customer == "Various"
