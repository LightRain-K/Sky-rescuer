"""Зонд модели на новых датасетах: оперативная метрика (тревога на кадре),
распределение уверенности, примеры с аннотацией. Без обучения."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "probe_flame"
OUT.mkdir(parents=True, exist_ok=True)

# класс fire в координатах каждого датасета -> наш класс fire=1
FIRE_CLS = {"flame1": 0, "essential_v3": 0}
# essential: 0=fire,1=smoke (порядок обратный нашему!) - для зонда смотрим как есть:
# наш детектор: 0=smoke, 1=fire
SMOKE_CLS = {"flame1": None, "essential_v3": 1}


def probe(m, ds, split, conf, save_n=6):
    imgdir = ROOT / "data" / ds / split / "images"
    labdir = ROOT / "data" / ds / split / "labels"
    imgs = sorted(imgdir.glob("*.jpg")) + sorted(imgdir.glob("*.png"))
    n_pos, hit, n_empty, fp_empty = 0, 0, 0, 0
    fire_confs, smoke_confs = [], []
    saved = 0
    for ip in imgs:
        lp = labdir / (ip.stem + ".txt")
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        gt_fire = any(int(p[0]) == FIRE_CLS[ds] for p in lines) if lines else False
        r = m.predict(str(ip), imgsz=960, conf=conf, verbose=False)[0]
        dets = [(int(b.cls), float(b.conf)) for b in res_boxes(r)]
        fmax = max([v for c, v in dets if c == 1], default=0.0)
        smax = max([v for c, v in dets if c == 0], default=0.0)
        fire_confs.append(fmax)
        smoke_confs.append(smax)
        if gt_fire:
            n_pos += 1
            hit += fmax > 0
        else:
            n_empty += 1
            fp_empty += (fmax > 0) or (smax > 0)
        if gt_fire and fmax == 0 and saved < save_n:
            r.save(filename=str(OUT / f"{ds}_{split}_{ip.stem}_miss.jpg"))
            saved += 1
    import numpy as np

    fa = np.array(fire_confs)
    print(
        f"{ds}/{split} conf={conf}: fire GT={n_pos} alarm={hit} "
        f"recall_img={hit / max(n_pos, 1):.3f} | empty={n_empty} fp={fp_empty} | "
        f"fireconf med={np.median(fa):.3f} p10={np.percentile(fa, 10):.3f}",
        flush=True,
    )


def res_boxes(r):
    return r.boxes if r.boxes is not None and len(r.boxes) else []


def main():
    m = YOLO(str(ROOT / "runs" / "mixed_v3_final" / "weights" / "best.pt"))
    for conf in (0.1, 0.05):
        probe(m, "flame1", "test", conf)
        probe(m, "essential_v3", "test", conf)
    # примеры попаданий для глаз
    for ds in ("flame1", "essential_v3"):
        ip = sorted((ROOT / "data" / ds / "test" / "images").glob("*"))[0]
        r = m.predict(str(ip), imgsz=960, conf=0.05, verbose=False)[0]
        r.save(filename=str(OUT / f"{ds}_sample0.jpg"))
    print("ann ->", OUT)


if __name__ == "__main__":
    main()
