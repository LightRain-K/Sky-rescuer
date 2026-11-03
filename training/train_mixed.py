"""Совместное обучение: D-Fire + свои кадры (апсемплинг custom x30).

Свои кадры повторяются 30 раз, чтобы их вес был ~4% батчей:
модель видит новый домен, но не забывает базу. Ранняя остановка
следит за D-Fire val — при забывании обучение встанет само.

Запуск:
    python training/train_mixed.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
REPEAT = 30


def build_expanded():
    src_img = ROOT / "data" / "custom" / "train" / "images"
    src_lab = ROOT / "data" / "custom" / "train" / "labels"
    dst = ROOT / "data" / "custom_x30" / "train"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    for img in sorted(src_img.glob("*.jpg")):
        lab = src_lab / (img.stem + ".txt")
        for i in range(REPEAT):
            name = f"{img.stem}_r{i:02d}"
            shutil.copy(img, dst / "images" / (name + ".jpg"))
            if lab.exists():
                shutil.copy(lab, dst / "labels" / (name + ".txt"))
    n = len(list((dst / "images").glob("*.jpg")))
    print(f"custom_x30: {n} images")


def main():
    build_expanded()
    yaml_path = ROOT / "data" / "mixed.yaml"
    yaml_path.write_text(
        "train:\n"
        "  - dfire/train/images\n"
        "  - custom_x30/train/images\n"
        "val: dfire/val/images\n"
        "\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    model = YOLO(str(ROOT / "runs" / "dfire_yolo11s_960" / "weights" / "best.pt"))
    model.train(
        data=str(yaml_path),
        epochs=15,
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=2,
        patience=6,
        project=str(ROOT / "runs"),
        name="mixed_dfire_custom",
    )
    print("--- D-Fire test ---")
    m = model.val(data=str(ROOT / "data" / "dfire.yaml"), split="test", imgsz=960, conf=0.1, verbose=False)
    print(f"mixed: fire P={m.box.p[1]:.3f} R={m.box.r[1]:.3f} mAP50={m.box.ap50[1]:.3f}")
    print("--- custom val ---")
    m2 = model.val(data=str(ROOT / "data" / "custom.yaml"), split="val", imgsz=960, conf=0.1, verbose=False)
    print(f"mixed on custom: all mAP50={m2.box.map50:.3f}")


if __name__ == "__main__":
    main()
