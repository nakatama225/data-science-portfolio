"""分析全体で共有する設定値。

「あとから変えたくなる値」と「変えると結果が変わる値」を、コードの中に
散らさず1か所に集めている。散らばっていると、片方だけ直して結果が
食い違ったことに気づけない。

値の由来（なぜこの値なのか）を各定数のコメントに残している。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Final

# --- 外部データの取得先 -----------------------------------------------------

SPACEX_API_ROOT: Final = "https://api.spacexdata.com/v4"
"""SpaceX が公開している REST API の入口。認証は不要。"""

PAST_LAUNCHES_ENDPOINT: Final = f"{SPACEX_API_ROOT}/launches/past"
"""過去の打ち上げ一覧。ロケットや積荷は ID だけが入っており、
詳細は別のエンドポイントを引き直す必要がある。"""

WIKIPEDIA_LAUNCH_LIST_URL: Final = (
    "https://en.wikipedia.org/w/index.php"
    "?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
)
"""Wikipedia の打ち上げ一覧。**版番号 (oldid) を固定している。**

最新版を指すと、ページが編集されるたびに表の構造が変わり、
同じコードが動かなくなったり、集計結果が静かに変わったりする。
分析をあとから再現できるようにするため、意図的に特定の版に固定している。
"""

# --- 分析の範囲 -------------------------------------------------------------

LAUNCH_DATE_CUTOFF: Final = dt.date(2020, 11, 13)
"""この日までの打ち上げだけを分析対象にする。

もう1つのデータ源である Wikipedia の版が、この日付までしか
記録していないため。2つのデータ源で対象期間を揃えないと、
突き合わせたときに片方にしかない打ち上げが混ざる。
"""

EXCLUDED_BOOSTER: Final = "Falcon 1"
"""分析から外すロケット。

Falcon 1 は初期の小型機で、機体の設計も運用も Falcon 9 とは別物。
着陸の成否を予測するという目的に対して、比較できる相手ではない。
"""

# --- 目的変数（予測したい対象）の定義 ---------------------------------------

UNSUCCESSFUL_LANDING_OUTCOMES: Final = frozenset(
    {
        "False ASDS",  # 洋上の無人船への着陸に失敗
        "False Ocean",  # 海面への軟着水に失敗
        "False RTLS",  # 発射場への帰還着陸に失敗
        "None ASDS",  # 洋上着陸を試みたが結果が記録されていない
        "None None",  # 着陸自体を試みていない
    }
)
"""「着陸に失敗した」とみなす結果コードの一覧。

元データの結果欄は `<成否> <着陸方式>` という2語の文字列になっている。
成否が `True` のものを成功、上記に挙げたものを失敗として扱う。

`None None`（そもそも着陸を試みていない）を失敗に含めているのは、
この分析の目的が「回収できるかどうか」の予測であり、
試みていない打ち上げは回収できていないため。
"""

# --- ファイルの置き場所 -----------------------------------------------------

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
"""このファイルから見たプロジェクトの最上位ディレクトリ。

`src/spacex_capstone/config.py` から2階層上がると
`ibm-capstone-spacex/` に着く。実行時の作業ディレクトリに依存しないよう、
ファイル自身の位置を起点にしている。
"""

DATA_DIR: Final = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final = DATA_DIR / "raw"
"""取得したままで、手を加えていないデータ。"""

PROCESSED_DATA_DIR: Final = DATA_DIR / "processed"
"""整形・加工した後のデータ。取得元から作り直せる。"""
