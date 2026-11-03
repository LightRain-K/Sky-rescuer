"""Разовая аналитика: распределение размеров fire-боксов по источникам train."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
dirs = [
    ROOT / "data" / "dfire" / "train" / "labels",
    ROOT / "data" / "merged_v3" / "custom_up" / "labels",
    ROOT / "data" / "merged_v3" / "afse" / "labels",
    ROOT / "data" / "merged_v3" / "web" / "labels",
]
for d in dirs:
    ws, hs = [], []
    fs = list(d.glob("*.txt")) if d.exists() else []
    for t in fs:
        for ln in t.read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) != 1:
                continue
            ws.append(float(p[3]))
            hs.append(float(p[4]))
    ws = np.array(ws)
    hs = np.array(hs)
    tiny = int(((ws < 0.025) & (hs < 0.04)).sum()) if len(ws) else 0
    med = f"{np.median(ws):.3f}/{np.median(hs):.3f}" if len(ws) else "-"
    print(f"{d.parent.name}: files={len(fs)} fire_boxes={len(ws)} tiny={tiny} median_wh={med}")
