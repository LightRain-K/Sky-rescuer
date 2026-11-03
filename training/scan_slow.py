"""Фоновый поиск тормозных файлов: timed predict по всем 4306 кадрам D-Fire test.

Печатает только файлы slower 1.0s. Порядок — как в списке, чтобы потом
сопоставить с позицией затыка в val.
"""
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

m = YOLO(str(ROOT / "runs" / "mixed_v3_final" / "weights" / "best.pt"))
imgs = sorted((ROOT / "data" / "dfire" / "test" / "images").glob("*.jpg"))
print("total:", len(imgs), flush=True)
slow = 0
t_all = time.time()
for i, p in enumerate(imgs):
    t = time.time()
    try:
        m.predict(str(p), imgsz=960, conf=0.1, verbose=False)
    except Exception as e:
        print(f"ERROR {i} {p.name}: {str(e)[:100]}", flush=True)
        continue
    dt = time.time() - t
    if dt > 1.0:
        slow += 1
        print(f"SLOW {i} {p.name}: {dt:.1f}s", flush=True)
print(f"done: {len(imgs)} files, {slow} slow, {time.time() - t_all:.0f}s total")
