"""Разведка новых датасетов: размеры, боксы, покрытие. Только статистика, без обучения."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

for ds in ("flame1", "essential_v3"):
    print(f"===== {ds} =====", flush=True)
    for split in ("train", "valid", "test"):
        imgdir = ROOT / "data" / ds / split / "images"
        labdir = ROOT / "data" / ds / split / "labels"
        imgs = sorted(imgdir.glob("*.jpg")) + sorted(imgdir.glob("*.png"))
        nbox, nempty, cls = 0, 0, set()
        ws, hs = [], []
        sizes = set()
        for ip in imgs[:400]:
            im = Image.open(ip)
            sizes.add(im.size)
        for ip in imgs:
            lp = labdir / (ip.stem + ".txt")
            if not lp.exists():
                nempty += 1
                continue
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                nempty += 1
                continue
            for p in lines:
                cls.add(int(p[0]))
                if len(p) >= 5:
                    ws.append(float(p[3]))
                    hs.append(float(p[4]))
                nbox += 1
        ws, hs = np.array(ws), np.array(hs)
        med = f"{np.median(ws):.3f}/{np.median(hs):.3f}" if len(ws) else "-"
        tiny = int(((ws < 0.025) & (hs < 0.04)).sum()) if len(ws) else 0
        print(
            f"{split}: imgs={len(imgs)} empty={nempty} boxes={nbox} "
            f"classes={sorted(cls)} median_wh={med} tiny={tiny} sizes={sorted(sizes)[:4]}",
            flush=True,
        )
