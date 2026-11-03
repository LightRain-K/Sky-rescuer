"""Дообучение модели на своих кадрах (вид сверху).

Старт с весов D-Fire, маленький learning rate, чтобы не забыть базу.
Валидация на своих 2 кадрах шумная — главный контроль по D-Fire test.

Запуск:
    python training/finetune.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    model = YOLO(str(ROOT / "runs" / "dfire_yolo11s_960" / "weights" / "best.pt"))
    model.train(
        data=str(ROOT / "data" / "custom.yaml"),
        epochs=30,
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=3,
        patience=10,
        project=str(ROOT / "runs"),
        name="finetune_custom",
    )
    print("--- D-Fire test (base weights) ---")
    base = YOLO(str(ROOT / "runs" / "dfire_yolo11s_960" / "weights" / "best.pt"))
    m0 = base.val(data=str(ROOT / "data" / "dfire.yaml"), split="test", imgsz=960, conf=0.1, verbose=False)
    print(f"base: fire R={m0.box.r[1]:.3f} mAP50={m0.box.ap50[1]:.3f}")
    print("--- D-Fire test (finetuned) ---")
    m1 = model.val(data=str(ROOT / "data" / "dfire.yaml"), split="test", imgsz=960, conf=0.1, verbose=False)
    print(f"finetuned: fire R={m1.box.r[1]:.3f} mAP50={m1.box.ap50[1]:.3f}")


if __name__ == "__main__":
    main()
