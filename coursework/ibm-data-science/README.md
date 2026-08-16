# IBM Data Science Professional Certificate

[IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science)
の課題（2023年8月提出）。

学習の履歴として保存しているもので、設計を練り直した成果物ではない。
この講座の最終課題を出発点に整理し直したものが
[ibm-capstone-spacex](../../ibm-capstone-spacex/) にある。

## 収録しているノートブック

| ファイル | 講座 | 内容 |
|---|---|---|
| [data-science-ecosystem.ipynb](notebooks/data-science-ecosystem.ipynb) | Tools for Data Science | データ分析で使う言語・ライブラリ・ツールの整理。Jupyter の基本操作 |
| [stock-data-analysis.ipynb](notebooks/stock-data-analysis.ipynb) | Python Project for Data Science | 株価と売上高の取得と可視化。API（yfinance）と Web ページ解析（BeautifulSoup）を組み合わせる |

## 注意

いずれも講座の実行環境で書いたもので、そのままでは手元で動かない箇所がある。

- `stock-data-analysis.ipynb` は `yfinance` を使って株価を取得する。
  外部サービスへの依存があり、サービス側の仕様変更で動かなくなることがある
- 出力セルは提出時のまま残してある。当時の実行結果を記録として保つため
