"""Обучение водной модели: YOLO11n на SeaDronesSee (люди в воде).

Отдельная модель под режим миссии «вода». Веса огня НЕ трогаем.
Holdout: официальный val (разметка есть; test без разметки).

Запуск из корня (сам следишь за процессом):
    python training/train_water.py [--epochs 40] [--imgsz 960] [--batch 16]

Старт с COCO-претрейна yolo11n.pt (скачается сам при первом запуске).
Выход: runs/sea_yolo11n_960/weights/best.pt
После: прогнать зонд и глазки на val, затем замерить swimmer alarm-rate.
Если пловец мелкий и recall низкий — следующий шаг imgsz=1280 или s-модель.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    from ultralytics import YOLO

    epochs, imgsz, batch = 40, 960, 16
    for a in sys.argv[1:]:
        if a.startswith("--epochs="):
            epochs = int(a.split("=", 1)[1])
        elif a.startswith("--imgsz="):
            imgsz = int(a.split("=", 1)[1])
        elif a.startswith("--batch="):
            batch = int(a.split("=", 1)[1])
        elif a == "--help":
            print(__doc__)
            return

    m = YOLO("yolo11n.pt")
    m.train(
        data=str(ROOT / "data" / "seax.yaml"),
        epochs=epochs, imgsz=imgsz, batch=batch,
        device=0, workers=4,
        project=str(ROOT / "runs"), name="sea_yolo11n_960",
        exist_ok=True,
    )
    print("best ->", ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
