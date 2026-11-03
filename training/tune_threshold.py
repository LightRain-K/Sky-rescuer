"""Подбор порога уверенности на тестовой выборке.

Запуск после обучения:
    python training/tune_threshold.py runs/dfire_yolo11s_960/weights/best.pt
"""
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/dfire_yolo11n/weights/best.pt")
    data_root = Path(__file__).resolve().parent.parent / "data" / "dfire"
    model = YOLO(str(weights))

    print(f"{'conf':>5} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}  (fire)")
    for conf in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        m = model.val(
            data=str(Path(__file__).resolve().parent.parent / "data" / "dfire.yaml"),
            split="test",
            imgsz=960,
            conf=conf,
            verbose=False,
        )
        # m.box: precision/recall/mAP50/mAP50-95 per class; class 1 = fire
        p = m.box.p[1]
        r = m.box.r[1]
        ap50 = m.box.ap50[1]
        ap = m.box.ap[1]
        print(f"{conf:5.2f} {p:6.3f} {r:6.3f} {ap50:7.3f} {ap:9.3f}")


if __name__ == "__main__":
    main()
