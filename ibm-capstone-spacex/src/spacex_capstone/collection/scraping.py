"""Wikipedia の打ち上げ一覧ページから記録を取り出す。

## なぜ Web ページから取るのか

SpaceX の Web API には、顧客名や打ち上げの成否（着陸ではなく、
打ち上げそのものの成否）が含まれていない。これらは Wikipedia の
一覧表にはあるので、2つのデータ源を併用する。

## HTML の表を読むときの難しさ

表のセルは、人が読むための飾りを含んでいる。

```html
<td>2010年6月4日<br/>18:45<sup>[12]</sup></td>
```

日付・時刻・脚注番号が1つのセルに同居していて、そのまま文字列にすると
`2010年6月4日18:45[12]` のように繋がってしまう。セルの種類ごとに
「どこを取り、どこを捨てるか」が違うため、セル1種類につき
1つの小さな関数を用意している。

関数を細かく分けているのは、**表の構造が変わったときに、
どの関数が壊れたかで原因の場所が分かる** ようにするため。
1つの大きな関数にまとめると、どのセルの読み方が変わったのか追えない。
"""

from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

LAUNCH_TABLE_CLASS = "wikitable plainrowheaders collapsible"
"""打ち上げ一覧の表に付いている CSS クラス。ページ内の他の表と区別するために使う。"""

_MASS_PATTERN = re.compile(r"([\d,]+(?:\.\d+)?)\s*kg")
"""`15,600 kg` のような表記から数値部分を取り出すための並び。

桁区切りのカンマと小数点の両方を許し、数値と `kg` の間の空白は
あってもなくてもよいものとして扱う。
"""


@dataclass(frozen=True, slots=True)
class ScrapedLaunch:
    """Wikipedia の表から取り出した打ち上げ1件分。

    数値に変換できるものは変換し、それ以外は表記のまま持つ。
    たとえば `flight_no` は整数だが、`orbit` は `LEO` のような記号なので
    文字列のままにしている。
    """

    flight_no: int
    date: str
    time: str
    booster_version: str
    launch_site: str
    payload: str
    payload_mass_kg: float | None
    orbit: str
    customer: str
    launch_outcome: str
    booster_landing: str


def parse_date_and_time(cell: Tag) -> tuple[str, str]:
    """日時のセルから、日付と時刻を分けて取り出す。

    このセルには日付・時刻に加えて脚注が入っていることがある。
    表示上は改行で区切られているため、区切りごとの文字列のうち
    先頭2つ（日付と時刻）だけを採用する。

    Returns:
        `("4 June 2010", "18:45")` のような組。
    """
    parts = [text.strip() for text in cell.strings]
    date = parts[0].strip(",") if parts else ""
    time = parts[1] if len(parts) > 1 else ""
    return date, time


def parse_booster_version(cell: Tag) -> str:
    """機体の型式と個体番号を取り出す（例: `F9 v1.0B0003.1`）。

    セルの中身は「型式・脚注・個体番号・脚注」の順に並ぶ。
    欲しいのは脚注以外の文字なので、**脚注を取り除いてから残りを繋ぐ**。

    Note:
        「奇数番目が脚注だから偶数番目だけ拾う」という数え方もできるが、
        その方法は末尾に改行が1つ入るかどうかで結果が変わる。
        HTML の空白は表示に影響しないため、ページの編集で簡単に増減する。
        脚注そのものを名指しで取り除くほうが、編集に強い。
    """
    working = copy.copy(cell)  # 呼び出し元の構造を壊さないよう複製してから触る
    for footnote in working.find_all("sup"):
        footnote.decompose()
    return "".join(working.stripped_strings)


def parse_landing_status(cell: Tag) -> str:
    """着陸の結果（`Success` / `Failure` など）を取り出す。

    セルには結果に続けて説明や脚注が入るため、先頭の語だけを採用する。
    """
    for text in cell.strings:
        stripped = text.strip()
        if stripped:
            return stripped
    return ""


def parse_payload_mass(cell: Tag) -> float | None:
    """積荷の重量をキログラム単位の数値として取り出す。

    元の表記は `15,600 kg (34,400 lb)` のように、桁区切りとポンド換算を
    伴う。全角文字や特殊な空白が混ざることもあるため、まず表記を
    標準的な形に揃えてから数値を探す。

    Returns:
        重量。重量の記載が無いセルでは `None`。

    Note:
        重量が無いときに 0 を返さないのは、**「積荷が 0 kg だった」と
        「重量が記録されていない」を区別するため**。0 を混ぜると平均値が
        実際より軽くなり、その後の欠損補完も効かなくなる。
    """
    normalised = unicodedata.normalize("NFKD", cell.text).strip()
    match = _MASS_PATTERN.search(normalised)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def extract_column_name(header_cell: Tag) -> str | None:
    """見出しセルから列名を取り出す。

    見出しには改行・リンク・脚注が含まれるため、それらを取り除いてから
    残った文字を列名とする。

    Returns:
        列名。数字だけの見出し（脚注番号など）と空の見出しは `None`。

    Note:
        複製してから飾りを取り除いている。元の要素を直接書き換えると、
        同じ表を別の目的でもう一度読んだときに、飾りが消えた状態の表を
        読むことになる。
    """
    working = copy.copy(header_cell)
    for decoration in ("br", "a", "sup"):
        found = getattr(working, decoration)
        if found:
            found.extract()

    name = " ".join(working.stripped_strings)
    if not name or name.isdigit():
        return None
    return name


class WikipediaLaunchTableParser:
    """打ち上げ一覧ページから `ScrapedLaunch` の並びを作る。

    Args:
        table_class: 対象とする表の CSS クラス。ページの改修で
            クラス名が変わったときに、コードを直さず差し替えられるよう
            引数にしている。
    """

    def __init__(self, table_class: str = LAUNCH_TABLE_CLASS) -> None:
        self.table_class = table_class

    def parse(self, soup: BeautifulSoup) -> list[ScrapedLaunch]:
        """ページ全体から、打ち上げ行だけを拾って一覧にする。"""
        launches: list[ScrapedLaunch] = []
        for table in soup.find_all("table", self.table_class):
            launches.extend(self._parse_table(table))
        return launches

    def _parse_table(self, table: Tag) -> Iterator[ScrapedLaunch]:
        for row in table.find_all("tr"):
            launch = self._parse_row(row)
            if launch is not None:
                yield launch

    def _parse_row(self, row: Tag) -> ScrapedLaunch | None:
        """1行を読む。打ち上げ行でなければ `None` を返す。

        打ち上げ行かどうかは **行頭の見出しセルが便名の数字かどうか** で
        判定する。表には小見出しや注釈の行が混ざっており、
        それらは行頭が数字にならない。
        """
        header = row.th
        if header is None or not header.string:
            return None
        flight_no = header.string.strip()
        if not flight_no.isdigit():
            return None

        cells = row.find_all("td")
        if len(cells) < 9:
            # 列が足りない行は、表の途中に挟まる注釈行。読み飛ばす。
            return None

        date, time = parse_date_and_time(cells[0])
        customer_link = cells[6].a

        return ScrapedLaunch(
            flight_no=int(flight_no),
            date=date,
            time=time,
            booster_version=parse_booster_version(cells[1]),
            launch_site=_first_link_text(cells[2]),
            payload=_first_link_text(cells[3]),
            payload_mass_kg=parse_payload_mass(cells[4]),
            orbit=_first_link_text(cells[5]),
            # 顧客が複数いる場合はリンクが張られていない。
            customer=customer_link.string if customer_link else "Various",
            launch_outcome=parse_landing_status(cells[7]),
            booster_landing=parse_landing_status(cells[8]),
        )


def _first_link_text(cell: Tag) -> str:
    """セル内の最初のリンクの文字列を返す。リンクが無ければセル全体の文字列。"""
    link = cell.a
    if link is not None and link.string:
        return link.string.strip()
    return cell.text.strip()
