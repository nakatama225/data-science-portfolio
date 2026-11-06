# Kaggle

Kaggle コンペティションへの参加記録と提出物。

## 一覧

| コンペ | 順位 / スコア | 内容 |
|---|---|---|
| [Deep Past Challenge - Translate Akkadian to English](deep-past-initiative-machine-translation/) | 276 位 / 2,674 チーム（35.8478） | アッカド語の翻字を英語に訳す。2つの ByT5 の訳の候補から、最小ベイズリスク（MBR）で1つを選ぶ推論。最終順位の元になった公開ノートブックのコードに、日本語の説明を付けた |

## ディレクトリの作り方

コンペごとに1つのディレクトリを作る。雛形が `_template/` にある。

```sh
cp -r kaggle/_template kaggle/<コンペ名>
```

コンペ名はハイフン区切りの小文字にする（Kaggle の URL の末尾に合わせる）。
例：`kaggle/titanic/`、`kaggle/house-prices-advanced-regression-techniques/`

作ったら、最上位の `pyproject.toml` の `[tool.uv.workspace] members` に
そのディレクトリを書き加える。これで、そのコンペ専用のライブラリを
他のプロジェクトに混ぜずに宣言できる。

## 書くときの約束

- **提出用のノートブックをそのまま置かない。** Kaggle のノートブックは
  1つのセルに全部入っていることが多い。処理は `src/` に切り出し、
  ノートブックは流れを追うためのものにする
- **順位とスコアを README に必ず書く。** 上位でなくてよい
- **提出ごとの点の一覧や振り返りは書かない。** README には、結果と、
  提出したものの説明だけを置く
- **データファイルをコミットしない。** Kaggle の規約上、再配布できないものがある。
  `kaggle/*/data/` はリポジトリの管理対象から外してあるので、
  手元にダウンロードしても入らない。取得手順は各コンペの README に書く

## データの扱い

コンペのデータは各自でダウンロードする。

```sh
# Kaggle API を使う場合
uv run kaggle competitions download -c <コンペ名> -p kaggle/<コンペ名>/data/
```
