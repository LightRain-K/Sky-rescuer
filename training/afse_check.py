"""Разовая проверка: GT-разметка AFSE (красный=fire, синий=smoke)
+ предсказания mixed-модели (зелёный) на одном кадре."""
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "afse_test_ours" / "images"
LAB_DIR = ROOT / "data" / "afse_test_ours" / "labels"

pick = None
for ip in sorted(IMG_DIR.glob("*.jpg")):
    lab = LAB_DIR / (ip.stem + ".txt")
    n = 0
    if lab.exists():
        n = sum(1 for ln in lab.read_text().splitlines() if ln.split()[0] == "1")
    if n >= 4:
        pick = ip
        break
print("pick:", pick.name)

im = Image.open(pick).convert("RGB")
W, H = im.size
d = ImageDraw.Draw(im)
lab = LAB_DIR / (pick.stem + ".txt")
gt_fire = 0
for ln in lab.read_text().splitlines():
    p = ln.split()
    c = int(p[0])
    x, y, w, h = map(float, p[1:5])
    x1, y1, x2, y2 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
    col = (255, 0, 0) if c == 1 else (0, 150, 255)
    d.rectangle([x1, y1, x2, y2], outline=col, width=3)
    d.text((x1 + 2, y1 + 2), "FIRE" if c == 1 else "SMOKE", fill=col)
    gt_fire += c == 1

m = YOLO(str(ROOT / "runs" / "mixed_dfire_custom" / "weights" / "best.pt"))
r = m.predict(str(pick), imgsz=960, conf=0.1, verbose=False)[0]
pred_fire = 0
if r.boxes is not None:
    for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
        d.rectangle(b, outline=(0, 255, 0), width=2)
        pred_fire += int(c) == 1

out = ROOT / "afse_check.jpg"
im.save(out)
print("GT fire:", gt_fire, "| pred fire:", pred_fire, "->", out)
