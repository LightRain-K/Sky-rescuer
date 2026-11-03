import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "dfire"
VAL_FRACTION = 0.1
SEED = 42


def main():
    train_imgs = sorted((ROOT / "train" / "images").glob("*.jpg"))
    random.Random(SEED).shuffle(train_imgs)
    n_val = int(len(train_imgs) * VAL_FRACTION)

    for split in ("images", "labels"):
        (ROOT / "val" / split).mkdir(parents=True, exist_ok=True)

    for img in train_imgs[:n_val]:
        label = ROOT / "train" / "labels" / (img.stem + ".txt")
        shutil.move(str(img), ROOT / "val" / "images" / img.name)
        shutil.move(str(label), ROOT / "val" / "labels" / label.name)

    print(f"val: {n_val}, train: {len(train_imgs) - n_val}")


if __name__ == "__main__":
    main()
