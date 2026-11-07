"""Подмешивание Pexels-кадров к SeaDronesSee train (рецепт как на огне).

Выход: data/sea_pex/pex_up/{images,labels} (pex x30) + data/seax_pex.yaml:
  train = 3 шарда SeaDronesSee + pex_up, val = официальный val (holdout).

Запуск: python training/build_sea_pex.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPEAT = 30


def main():
    src = ROOT / "data" / "pex_labels"
    out = ROOT / "data" / "sea_pex" / "pex_up"
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    imgs = sorted(src.glob("*.jpg"))
    n = 0
    for ip in imgs:
        lp = src / (ip.stem + ".txt")
        if not lp.exists():
            print("нет разметки:", ip.name)
            continue
        for k in range(REPEAT):
            shutil.copy(ip, out / "images" / f"pex_{ip.stem}_{k:02d}.jpg")
            shutil.copy(lp, out / "labels" / f"pex_{ip.stem}_{k:02d}.txt")
            n += 1
    (ROOT / "data" / "seax_pex.yaml").write_text(
        "train:\n"
        "  - seadronessee/data/images/train/shard_000\n"
        "  - seadronessee/data/images/train/shard_001\n"
        "  - seadronessee/data/images/train/shard_002\n"
        "  - sea_pex/pex_up/images\n"
        "val: seadronessee/data/images/val/shard_000\n"
        "nc: 5\n"
        "names:\n"
        "  0: swimmer\n"
        "  1: boat\n"
        "  2: jetski\n"
        "  3: life_saving_appliances\n"
        "  4: buoy\n"
    )
    print(f"pex {len(imgs)} кадров x{REPEAT} = {n} + SeaDronesSee 8930")


if __name__ == "__main__":
    main()
