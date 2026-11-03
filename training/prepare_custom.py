"""Подготовка своих размеченных данных к дообучению.

Ожидает папку, где лежат .jpg и .txt в формате YOLO с теми же классами,
что в D-Fire (0 = smoke, 1 = fire). Пустые .txt или их отсутствие = фон.
Раскладывает в data/custom/{train,val}/{images,labels} (90/10, seed 42)
и пишет data/custom.yaml.

Запуск:
    python training/prepare_custom.py data/custom_labeled
"""
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "custom_labeled"
    dst = ROOT / "data" / "custom"
    imgs = sorted([p for p in src.glob("*.jpg") if (src / (p.stem + ".txt")).exists()
                   or True])  # фон без txt тоже годится
    imgs = sorted(src.glob("*.jpg"))
    if not imgs:
        print(f"В {src} нет .jpg")
        return

    random.Random(SEED).shuffle(imgs)
    n_val = max(1, int(len(imgs) * 0.1))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}

    for split, files in splits.items():
        for sub in ("images", "labels"):
            (dst / split / sub).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, dst / split / "images" / img.name)
            label = src / (img.stem + ".txt")
            if label.exists():
                shutil.copy(label, dst / split / "labels" / label.name)
            else:
                (dst / split / "labels" / (img.stem + ".txt")).write_text("")

    (ROOT / "data" / "custom.yaml").write_text(
        "train: custom/train/images\n"
        "val: custom/val/images\n"
        "\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    print(f"train: {len(splits['train'])}, val: {len(splits['val'])} -> {dst}")


if __name__ == "__main__":
    main()
