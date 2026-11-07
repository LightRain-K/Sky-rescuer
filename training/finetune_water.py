"""Дообучение водной модели на Pexels-кадрах (крупные пловцы).

Старт с runs/sea_yolo11n_960/weights/best.pt, малый LR, заморозка спины —
чтобы не забыть SeaDronesSee. Val-holdout не трогаем.
Перед запуском: python training/build_sea_pex.py

Запуск: python training/finetune_water.py [--epochs 15]
Выход: runs/sea_v2_pex/weights/best.pt -> проверка eval_water.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    from ultralytics import YOLO

    epochs = 15
    for a in sys.argv[1:]:
        if a.startswith("--epochs="):
            epochs = int(a.split("=", 1)[1])
        elif a == "--help":
            print(__doc__)
            return
    m = YOLO(str(ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt"))
    m.train(
        data=str(ROOT / "data" / "seax_pex.yaml"),
        epochs=epochs, imgsz=960, batch=16,
        lr0=0.002, warmup_epochs=1, freeze=10,
        device=0, workers=4,
        project=str(ROOT / "runs"), name="sea_v2_pex",
        exist_ok=True,
    )
    print("best ->", ROOT / "runs" / "sea_v2_pex" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
