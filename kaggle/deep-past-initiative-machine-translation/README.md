# Deep Past Challenge - Translate Akkadian to English

- **コンペのページ**：<https://www.kaggle.com/competitions/deep-past-initiative-machine-translation>
- **期間**：2025-12-16 〜 2026-03-23
- **結果**：276 位 / 2,674 チーム（非公開の点 35.8478・評価指標は BLEU と chrF++ の幾何平均）。
  ブロンズ（上位 10%、267 位まで）に 9 位届かなかった

このフォルダにあるのは、最終順位の元になった提出のノートブック。
**コードは公開ノートブックと同じもので、自分で書いたものではない。**
このリポジトリで足したのは日本語の説明（英語のコメントの訳と、節ごとの解説）だけ。
自分で試したことと、その点は、下の「取り組みの記録」に書いた。

---

## Why — 何を解くコンペか

約 4000 年前のアッシリアの商人が粘土板に残した手紙・送り状・契約の文（古アッシリア語。
アッカド語の古い方言）を、英語に訳すモデルを作る。主催者によると、粘土板は約 2 万 3 千枚が
残り、訳されたのは半分ほどで、残りを読める専門家は世界に 12 人もいない。

入力は、楔形文字をラテン文字に写した翻字（transliteration）で、出力は英語の訳文。
学習に使える対訳が少なく、アッカド語は英語の数語ぶんの意味を1語の語形変化で表すので、
データの多い現代の言語向けの作り方がそのままでは通じない、と主催者は書いている。

**評価指標**：BLEU と chrF++ の幾何平均。どちらも、文ごとの点の平均ではなく、コーパス全体で
集計して求める。BLEU は語の n-gram の一致を、chrF++ は文字の n-gram と、語の 2-gram までの
一致を見るので、語の選び方と綴りの両方が点に影響する。

**提出の条件**：Kaggle のノートブックで提出する。GPU で 9 時間以内、インターネットは使えない。
公開されている学習済みのモデルは使ってよい。

## Design Choice — このノートブックの方針

- 学習はせず、公開されている2つの ByT5 の学習済みモデルで推論だけをする
- 2つのモデルから、1つの文につき 10 個ずつ訳の候補を作る（ビームサーチの上位 4 個と、
  温度を変えたサンプリングの 6 個）。合わせた 20 個から、ほかの候補と最も似ているものを選ぶ
  （最小ベイズリスク〔MBR〕による選択）。似ている度合いは、chrF++・BLEU・Jaccard の重み付きの
  平均に、長さの点を足したもの
- 前処理で翻字の表記の揺れを、後処理で訳の書式の揺れを揃える。後処理には、主催者がデータの
  書き方を更新したのに合わせた規則（シェケルの端数・丸かっこ・引用符など）を含む

詳しくは、ノートブックの各節の説明に書いた。

## 出典とライセンス

| | 作者 | ページ |
|---|---|---|
| コード | Vitor Hugo Barbedo「LB 35.9 with Regex corrections (Public Model)」の版5（このときの題は「LB 35.6 com correções」） | <https://www.kaggle.com/code/vitorhugobarbedo/lb-35-9-with-regex-corrections-public-model> |
| コードの元 | Cho Royou「Better Candidate Diversity on Public Model」。上のノートブックは、これに後処理の修正7件を加えたもの | <https://www.kaggle.com/code/ngyzly/better-candidate-diversity-on-public-model> |
| モデル A | Assia Benkedia「byt5-akkadian-optimized-34x」（データセット assiaben/final-byt5） | <https://www.kaggle.com/datasets/assiaben/final-byt5> |
| モデル B | Mattia Angeli「byt5-akkadian-mbr-v2」 | <https://www.kaggle.com/models/mattiaangeli/byt5-akkadian-mbr-v2> |

- コード（Vitor Hugo Barbedo のノートブック）は、ページの License の欄によると Apache License 2.0
  で公開されている。全文は、このフォルダの [LICENSE](LICENSE)
- 変えた所：英語のコメントを日本語の説明に置き換え、1つだったセルを節ごとに分け、先頭の説明の
  文字列（docstring）を最初のマークダウンのセルに訳して移した。コードは変えていない。コードが
  元と同じことは、[tests/test_notebook_code.py](tests/test_notebook_code.py) が構文木を比べて確認する
- モデルの重みはリポジトリに含めない。Kaggle の上で、ノートブックの入力に追加して使う

## 取り組みの記録

提出したときに付けた説明と、そのときの非公開の点。コンペが終わった後に、Kaggle の提出の一覧から
写した。

| 日付 | 試したこと | 非公開の点 |
|---|---|---|
| 1/30〜1/31 | 公開ノートブック（ByT5 の推論）をそのまま提出 | 33.77 |
| 2/6〜2/7 | ByT5-small に LoRA を付け、手元の Mac（MPS）で学習。5 エポックと 30 エポック | 2.17 / 8.56 |
| 2/9 | 推論の出力に、辞書と分野の規則（銀の表記・度量衡）を足す | 8.56〜33.53 |
| 2/10 | 入力の頭書き「translate Akkadian to English: 」を付けない版と、付けた版 | 18.57 / 33.15 |
| 2/11〜2/14 | 公開の推論（Chunky v1.5.0。そのまま動かして 33.56）で、4 万 9 千語の辞書の訳を出力に差し込む方法・ビームの幅・長さの罰則・入出力の最大の長さを変える | 18.62〜33.56。元の 33.56 を超えなかった |
| 2/19 | 公開の推論（LB 35.2）を、主催者が更新した欠けた所の印（`<gap>`）の書き方に合わせる | 33.98 |
| 3/17 | このノートブック | **35.85** |
| 3/21 | 公開ノートブック「Hybrid best Akkadian」 | 35.24 |
| 3/23 | 公開ノートブックの別の版と、2つのモデルの候補を MBR で選ぶ処理を短く書き直した版 | 1 未満（原因は確認していない） |

## 振り返り

- **点が上がったこと**：使う公開ノートブックを替えたとき（33.77 → 33.98 → 35.85）。
  入力の頭書きを付けたとき（18.57 → 33.15）
- **点が上がらなかったこと**：辞書の訳を出力に後から差し込む（元の 33.56 を超えなかった）。
  手元の Mac での LoRA の学習（8.56。公開の学習済みモデルの 33 前後に届かなかった）

## How-to — 動かし方

Kaggle の上で動かす（手元では動かさない）。

1. Kaggle の新しいノートブックに、`notebooks/01_ensemble-mbr-inference.ipynb` を読み込む
2. 入力に、コンペのデータ、データセット `assiaben/final-byt5`、モデル
   `mattiaangeli/byt5-akkadian-mbr-v2`（PyTorch・default・版 1）を追加する
3. アクセラレータを GPU（T4）にし、インターネットを切って実行する。`/kaggle/working/submission.csv` ができる

手元では、ノートブックのコードが元と同じことだけを検査する（リポジトリの最上位で実行する）。

```sh
uv run pytest kaggle/deep-past-initiative-machine-translation/tests
```

## 構成

```text
notebooks/  提出したノートブック（日本語の説明つき）
tests/      ノートブックのコードが元の公開ノートブックと同じことの検査
LICENSE     元のノートブックのライセンス（Apache License 2.0）の全文
```

雛形（`_template/`）にある `pyproject.toml`・`src/`・`data/` は置いていない。推論は Kaggle の
GPU の上で動かすもので、手元で動かす部品とデータが無いため。検査は標準ライブラリだけで動くので、
ワークスペースにも加えていない。
