"""ノートブックのコードが、元にした公開ノートブックのコードと同じことの検査。

ノートブックには日本語の説明を足し、1つだったセルを節ごとに分けたが、コードは
変えていない。説明を書き足すうちにコードの字を消したり足したりしても、ノートブックは
そのまま開けるので、Kaggle で動かすまで気づけない。そこで、コードの構文木を元と比べる。

コメント・空白・改行・セルの分け方は構文木に現れないので、説明をどう書き換えても
結果は変わらない。元の先頭にあった説明の文字列（docstring）は、最初のマークダウンの
セルに訳して移したので、比べる対象から外してある。
"""

import ast
import hashlib
import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "01_ensemble-mbr-inference.ipynb"
)

# 元の公開ノートブック（版5）のコードから先頭の docstring を除き、下の code_digest と
# 同じ求め方で求めた値。Python 3.11 と 3.14 で同じ値になる。
ORIGINAL_DIGEST = "e57faffb8504db8a6ab2f1ac9a8e98e3e4f9be78b49796d211d45f14f0e54b56"


def load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> str:
    """コードのセルを、上から順に空行を挟んでつなぐ。"""
    return "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def canonical(node: object) -> object:
    """構文木を、Python の版で変わらない入れ子の tuple にする。

    ``ast.dump`` の出力は版で変わる（3.13 から、値が None や空の欄を省く）。版で
    欄が増えることもある（3.12 で関数とクラスに ``type_params`` が増えた）。そこで、
    値が None か空の並びの欄は省き、残りの欄を名前と一緒に並べる。行と列の位置は
    ``_fields`` に含まれないので、比べる対象にならない。
    """
    if isinstance(node, ast.AST):
        fields = []
        for name in node._fields:
            value = getattr(node, name, None)
            if value is None or value == []:
                continue
            fields.append((name, canonical(value)))
        return (type(node).__name__, tuple(fields))
    if isinstance(node, list):
        return tuple(canonical(item) for item in node)
    return node


def code_digest(source: str) -> str:
    """コードの構文木から求めた sha256。"""
    tree = ast.parse(source)
    return hashlib.sha256(repr(canonical(tree)).encode("utf-8")).hexdigest()


def test_ノートブックのコードは元の公開ノートブックと同じ() -> None:
    assert code_digest(code_cells(load())) == ORIGINAL_DIGEST


def test_コードの数値を1つ変えると_一致しなくなる() -> None:
    """比べ方が何を渡しても同じ値を返すと、上のテストは何も見ずに通る。"""
    source = code_cells(load())
    changed = source.replace(
        "mbr_w_chrf:     float = 0.55", "mbr_w_chrf:     float = 0.56"
    )
    assert changed != source
    assert code_digest(changed) != ORIGINAL_DIGEST


def test_コメントと空行を足しても_一致したまま() -> None:
    """説明を書き換えるたびに、固定値を求め直さなくてよい。"""
    source = code_cells(load())
    assert code_digest("# 説明\n\n" + source + "\n# 末尾の説明\n") == ORIGINAL_DIGEST


def test_コードのセルに実行の出力を残さない() -> None:
    """実行した後に保存し直すと、テストデータの id と、それに対するモデルの訳が残る。

    コンペのデータに由来するものはリポジトリに置かない（kaggle/README.md）ので、
    出力を残したままのノートブックを置いたときに止める。
    """
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
