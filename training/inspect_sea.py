"""Разведка SeaDronesSee val: классы, размеры боксов, разрешения."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "seadronessee" / "data"

print((D / "data.yaml").read_text())
imgs = sorted((D / "images" / "val").rglob("*.jpg"))
print("val images:", len(imgs))

sizes = {}
for ip in imgs[:200]:
    sizes[Image.open(ip).size] = sizes.get(Image.open(ip).size, 0) + 1
print("sizes(sample200):", sizes)

boxes = {}
for ip in imgs:
    lp = D / "labels" / "val" / "shard_000" / (ip.stem + ".txt")
    if not lp.exists():
        continue
    for ln in lp.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        boxes.setdefault(int(p[0]), []).append((float(p[3]), float(p[4])))

names = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
for c in sorted(boxes):
    a = np.array(boxes[c])
    tiny = ((a[:, 0] < 0.03) & (a[:, 1] < 0.03)).mean() * 100
    print(f"{names[c]}: n={len(a)} median_wh={np.median(a[:,0]):.4f}/{np.median(a[:,1]):.4f} "
          f"tiny(<3% кадра)={tiny:.0f}% max_wh={a.max(axis=0).round(3)}")
