# Deep Past Challenge - Translate Akkadian to English

- コンペのページ：<https://www.kaggle.com/competitions/deep-past-initiative-machine-translation>
- 期間：2025-12-16 〜 2026-03-23
- 結果：276 位 / 2,674 チーム（非公開の点 35.8478・評価指標は BLEU と chrF++ の幾何平均）。
  ブロンズ（上位 10%、267 位まで）に 9 位届かなかった

このフォルダには、最終順位の元になった提出のノートブックを置いた。
コードは公開ノートブックと同じもので、自分で書いたものではない。
このリポジトリでは、日本語の説明（英語のコメントの訳と、節ごとの解説）だけを足した。

---

## 何を解くコンペか

約 4000 年前のアッシリアの商人が粘土板に残した手紙・送り状・契約の文（古アッシリア語。
アッカド語の古い方言）を、英語に訳すモデルを作る。主催者によると、粘土板は約 2 万 3 千枚が
残り、訳されたのは半分ほどで、残りを読める専門家は世界に 12 人もいない。

入力は楔形文字をラテン文字に写した翻字（transliteration）で、出力は英語の訳文。
学習に使える対訳が少なく、アッカド語は英語の数語ぶんの意味を1語の語形変化で表すので、
データの多い現代の言語向けの作り方がそのままでは通じない、と主催者は書いている。

評価指標は BLEU と chrF++ の幾何平均で、どちらも文ごとの点の平均ではなく、コーパス全体で
集計して求める。BLEU は語の n-gram の一致を、chrF++ は文字の n-gram と、語の 2-gram までの
一致を見るので、語の選び方と綴りの両方が点に影響する。

提出には Kaggle のノートブックを使う。ノートブックの実行は GPU で 9 時間以内で、インターネットは
使えない。公開されている学習済みのモデルは使ってよい。

## このノートブックの方針

- 学習はせず、公開されている2つの ByT5 の学習済みモデルで推論だけをする
- 2つのモデルから、1つの文につき 10 個ずつ訳の候補を作る（ビームサーチの上位 4 個と、
  温度を変えたサンプリングの 6 個）。合わせた 20 個から、ほかの候補と最も似ているものを選ぶ
  （最小ベイズリスク〔MBR〕による選択）。似ている度合いは、chrF++・BLEU・Jaccard の重み付きの
  平均に、長さの点を足したもの
- 前処理で翻字の表記の揺れを、後処理で訳の書式の揺れを揃える。後処理は、主催者がデータの
  書き方を更新したのに合わせた規則（シェケルの端数・丸かっこ・引用符など）を含む

詳しくはノートブックの各節に書いた。

## 出典とライセンス

| | 作者 | ページ |
|---|---|---|
| コード | Vitor Hugo Barbedo「LB 35.9 with Regex corrections (Public Model)」の版5（このときの題は「LB 35.6 com correções」） | <https://www.kaggle.com/code/vitorhugobarbedo/lb-35-9-with-regex-corrections-public-model> |
| コードの元 | Cho Royou「Better Candidate Diversity on Public Model」。上のノートブックは、これに後処理の修正7件を加えたもの | <https://www.kaggle.com/code/ngyzly/better-candidate-diversity-on-public-model> |
| モデル A | Assia Benkedia「byt5-akkadian-optimized-34x」（データセット assiaben/final-byt5） | <https://www.kaggle.com/datasets/assiaben/final-byt5> |
| モデル B | Mattia Angeli「byt5-akkadian-mbr-v2」 | <https://www.kaggle.com/models/mattiaangeli/byt5-akkadian-mbr-v2> |

- 表の「コード」と「コードの元」の2つのノートブックは、どちらもページの License の欄に
  Apache License 2.0 と書いてある。全文はこのフォルダの [LICENSE](LICENSE) に置いた
- このリポジトリでは、英語のコメントを日本語の説明に置き換え、1つだったセルを節ごとに分け、
  先頭の説明の文字列（docstring）を最初のマークダウンのセルに訳して移した。コードは変えていない。
  コードが元と同じことは、[tests/test_notebook_code.py](tests/test_notebook_code.py) が構文木を比べて確認する
- モデルの重みはリポジトリに含めない。Kaggle の上で、ノートブックの入力に追加して使う

## 動かし方

Kaggle の上で動かす（手元では動かさない）。

1. Kaggle の新しいノートブックに、`notebooks/01_ensemble-mbr-inference.ipynb` を読み込む
2. 入力に、コンペのデータ、データセット `assiaben/final-byt5`、モデル
   `mattiaangeli/byt5-akkadian-mbr-v2`（PyTorch・default・版 1）を追加する
3. アクセラレータを GPU（T4）にし、インターネットを切って実行する。`/kaggle/working/submission.csv` ができる

手元では、ノートブックのコードが元と同じことだけを検査する。リポジトリの最上位で次を実行する。

```sh
uv run pytest kaggle/deep-past-initiative-machine-translation/tests
```

## 構成

```text
notebooks/  提出したノートブック（日本語の説明つき）
tests/      ノートブックのコードが元の公開ノートブックと同じことの検査
LICENSE     元のノートブックのライセンス（Apache License 2.0）の全文
```

推論は Kaggle の GPU の上で動かし、手元で動かす部品もデータも無いので、雛形（`_template/`）にある
`pyproject.toml`・`src/`・`data/` は置いていない。検査は標準ライブラリだけで動くので、
ワークスペースにも加えていない。
