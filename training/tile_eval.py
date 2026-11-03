"""Оценка recall по размерам очагов огня с тайлингом (SAHI-подобный подход).

Запуск:
    python training/tile_eval.py runs/dfire_yolo11s_960/weights/best.pt
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

root = Path(__file__).resolve().parent.parent
data = root / "data" / "dfire"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0


def predict_tiled(model, img, tile=640, overlap=0.25, conf=0.1):
    W, H = img.size
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    preds = []
    for y0 in ys:
        for x0 in xs:
            crop = img.crop((x0, y0, x0 + tile, y0 + tile))
            r = model.predict(crop, imgsz=tile, conf=conf, verbose=False)[0]
            if r.boxes is None:
                continue
            for b, cls, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(cls) != 1:
                    continue
                x1, y1, x2, y2 = b
                preds.append(((x1 + x0) / W, (y1 + y0) / H, (x2 + x0) / W, (y2 + y0) / H, float(cf)))
    # NMS по IoU
    kept = []
    preds.sort(key=lambda p: -p[4])
    for p in preds:
        if all(iou(p[:4], k[:4]) < 0.5 for k in kept):
            kept.append(p)
    return kept


def bucket(area):
    if area < 0.003:
        return "tiny<0.3%"
    if area < 0.01:
        return "small0.3-1%"
    if area < 0.05:
        return "med1-5%"
    return "large>5%"


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "runs/dfire_yolo11n/weights/best.pt"
    model = YOLO(str(weights))

    stems = []
    for p in (data / "test" / "labels").glob("*.txt"):
        t = p.read_text().strip()
        if any(l.split()[0] == "1" for l in t.splitlines() if l.strip()):
            stems.append(p.stem)
    random.Random(0).shuffle(stems)
    stems = stems[:200]

    bins = {k: [] for k in ("tiny<0.3%", "small0.3-1%", "med1-5%", "large>5%")}
    n_total = 0
    for stem in stems:
        ip = data / "test" / "images" / (stem + ".jpg")
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        gts = []
        for ln in (data / "test" / "labels" / (stem + ".txt")).read_text().splitlines():
            c, x, y, w, h = (float(v) for v in ln.split())
            if int(c) == 1:
                gts.append(((x - w / 2, y - h / 2, x + w / 2, y + h / 2), w * h))
        preds = predict_tiled(model, img, tile=640, overlap=0.25, conf=0.1)
        for gt, area in gts:
            hit = any(iou(gt, p[:4]) >= 0.5 for p in preds)
            bins[bucket(area)].append(1 if hit else 0)
            n_total += 1

    print("Tiled (640, overlap 0.25, conf=0.1):")
    for k, v in bins.items():
        v = np.array(v)
        print(f"  {k}: n={len(v)} recall={v.mean():.3f}" if len(v) else f"  {k}: n=0")
    allv = np.concatenate([np.array(v) for v in bins.values() if len(v)])
    print(f"  ALL fire: n={len(allv)} recall={allv.mean():.3f}")


if __name__ == "__main__":
    main()
