"""Подготовка 'фото пожары' к разметке: jpg-конвертация, чистка дублей, предразметка.

Берёт newest веса (mixed_v2 -> mixed_dfire_custom), складывает всё в
data/web_work/images (кадр + txt рядом + classes.txt) — дальше labelImg.

Запуск:
    python training/prepare_web.py
"""
import hashlib
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "фото пожары"
DST = ROOT / "data" / "web_work" / "images"


def best_weights() -> Path:
    for cand in ("runs/mixed_v2/weights/best.pt",
                 "runs/mixed_dfire_custom/weights/best.pt"):
        if (ROOT / cand).exists():
            return ROOT / cand
    raise FileNotFoundError("нет весов mixed_v2/mixed_dfire_custom")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    kept: list[Path] = []
    for f in sorted(SRC.iterdir()):
        if not f.is_file():
            continue
        try:
            im = Image.open(f)
            im.load()
        except Exception as e:
            print("SKIP unreadable:", f.name, str(e)[:60])
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            print("SKIP dupe:", f.name, "== ", seen[h])
            continue
        seen[h] = f.name
        dst_img = DST / (f.stem + ".jpg")
        i = 1
        while dst_img.exists():
            dst_img = DST / f"{f.stem}_{i}.jpg"
            i += 1
        Image.open(f).convert("RGB").save(dst_img, quality=92)
        kept.append(dst_img)
    (DST / "classes.txt").write_text("smoke\nfire\n")
    print(f"images ready: {len(kept)}")

    w = best_weights()
    print("weights:", w)
    m = YOLO(str(w))
    m.predict(str(DST), imgsz=960, conf=0.1, save_txt=True, save_conf=True,
              project=str(ROOT / "runs"), name="prelabel_web", exist_ok=True)
    # разложить txt рядом с кадрами, отрезать confidence-колонку
    lab_src = ROOT / "runs" / "prelabel_web" / "labels"
    n_lab = 0
    for img in DST.glob("*.jpg"):
        lab = lab_src / (img.stem + ".txt")
        if lab.exists():
            lines = []
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if len(p) >= 5:
                    lines.append(" ".join(p[:5]))
            (DST / (img.stem + ".txt")).write_text(
                "\n".join(lines) + "\n" if lines else "")
            n_lab += 1
        else:
            (DST / (img.stem + ".txt")).write_text("")
    print(f"prelabels placed: {n_lab}/{len(kept)} -> {DST}")


if __name__ == "__main__":
    main()
