"""Путаница swimmer->boat/jetski на SeaDronesSee val: считаем GT-пловцов,
перекрытых предсказанными лодками/гидроциклами (IoU>0.45)."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def main(conf=0.25):
    m = YOLO(str(ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt"))
    d = ROOT / "data" / "seadronessee" / "data"
    imgs = sorted((d / "images" / "val").rglob("*.jpg"))
    n_sw, confused, confs, areas, ex = 0, 0, [], [], []
    for ip in imgs:
        lp = d / "labels" / "val" / "shard_000" / (ip.stem + ".txt")
        gt = []
        for ln in lp.read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) != 0:
                continue
            x, y, w, h = map(float, p[1:5])
            gt.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2, w * h))
        n_sw += len(gt)
        r = m.predict(str(ip), imgsz=960, conf=conf, verbose=False)[0]
        pred = [(b.xyxy.tolist()[0], int(b.cls), float(b.conf))
                for b in (r.boxes or []) if int(b.cls) in (1, 2)]
        for g in gt:
            best = max([iou(g[:4], pb) for pb, _, _ in pred], default=0.0)
            if best > 0.45:
                hit = max([(iou(g[:4], pb), pc, pv) for pb, pc, pv in pred])[1:]
                confused += 1
                confs.append(hit[1])
                areas.append(g[4])
                if len(ex) < 8:
                    ex.append((ip.name, hit[0], round(hit[1], 2), round(g[4], 4)))
    import numpy as np
    print(f"swimmer GT: {n_sw}, названы boat/jetski: {confused} ({100 * confused / max(n_sw, 1):.1f}%)")
    if confs:
        print(f"conf медиана: {np.median(confs):.2f}, площадь GT медиана: {np.median(areas):.4f} "
              f"(~{int((np.median(areas) * 3840 * 2160) ** 0.5)}px сторона на 4K)")
        for e in ex:
            print("  ", e)


if __name__ == "__main__":
    main()
