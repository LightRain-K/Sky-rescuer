"""Сборка merged v3: D-Fire + custom (x30) + AFSE train/valid (x5, свап) + web (x8).

Web: data/web_work/images — свои фото из интернета, классы уже наши.
AFSE test НЕ трогаем — вечный holdout (data/afse_test_ours).

Выход:
  data/merged_v3/custom_up, data/merged_v3/afse, data/merged_v3/web
  data/merged_v3.yaml

Запуск:
    python training/build_mixed_v3.py
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mixed_v2 import materialize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "merged_v3"


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    stat: Counter = Counter()

    n_custom = materialize(
        ROOT / "data" / "custom" / "train" / "images",
        ROOT / "data" / "custom" / "train" / "labels",
        OUT / "custom_up", "custom", 30, False, stat,
    )
    n_afse = 0
    for split in ("train", "valid"):
        n_afse += materialize(
            ROOT / "data" / "afse" / split / "images",
            ROOT / "data" / "afse" / split / "labels",
            OUT / "afse", "afse", 5, True, stat,
        )
    n_web = materialize(
        ROOT / "data" / "web_work" / "images",
        ROOT / "data" / "web_work" / "images",
        OUT / "web", "web", 8, False, stat,
    )
    n_dfire = len(list((ROOT / "data" / "dfire" / "train" / "images").glob("*.jpg")))

    (ROOT / "data" / "merged_v3.yaml").write_text(
        "train:\n"
        "  - dfire/train/images\n"
        "  - merged_v3/custom_up/images\n"
        "  - merged_v3/afse/images\n"
        "  - merged_v3/web/images\n"
        "val: dfire/val/images\n"
        "\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    total = n_dfire + n_custom + n_afse + n_web
    print(f"dfire train : {n_dfire} ({100 * n_dfire / total:.1f}%)")
    print(f"custom x30  : {n_custom} ({100 * n_custom / total:.1f}%)")
    print(f"afse   x5   : {n_afse} ({100 * n_afse / total:.1f}%)")
    print(f"web    x8   : {n_web} ({100 * n_web / total:.1f}%)")
    print(f"total train : {total}")
    print("boxes:", dict(stat))
    print("yaml: data/merged_v3.yaml")


if __name__ == "__main__":
    main()
