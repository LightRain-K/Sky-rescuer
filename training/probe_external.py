"""Внешний зонд: кадры, которых нет в обучении (Wikimedia Commons, свободные
лицензии USFS/USGS/NASA), + кроп-зонды пропущенного кадра AFSE.

Выход: data/probe_external/*.jpg (исходники + аннотации), печать детекций.
Папка НЕ входит ни в один train-yaml - только для проверки.
"""
import urllib.request
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "probe_external"
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    # вид с дрона пожарной команды над мозаичным выжиганием (ожидаем дым/огонь)
    "burn_mosaic_drone.jpg": "https://thumb.wikimedia.org/wikipedia/commons/thumb/d/d2/This_drone%2C_flown_by_the_fire_crew%2C_gives_them_an_overhead_view_of_the_burn_area._The_mosaic_pattern_of_burning_is_intended_to_%28a3cbb8f2-5f42-445d-88d5-c6d31a5c48fc%29.JPG/1280px-thumbnail.jpg",
    # плановое выжигание Inyo NF (ожидаем дым)
    "inyo_prescribed.jpg": "https://thumb.wikimedia.org/wikipedia/commons/thumb/2/2e/Inyo_National_Forest_2023_Prescribed_Burns_%2820230625-FS-FS-041%29.jpg/1280px-Inyo_National_Forest_2023_Prescribed_Burns_%2820230625-FS-FS-041%29.jpg",
    # лес с дрона БЕЗ огня - жесткие негативы (ждем тишину)
    "usgs_forest_neg1.jpg": "https://thumb.wikimedia.org/wikipedia/commons/thumb/b/bd/USGS_Team_Flies_Drones_for_Oregon_BLM_Forestry_Research_%2814594308049%29.jpg/1280px-USGS_Team_Flies_Drones_for_Oregon_BLM_Forestry_Research_%2814594308049%29.jpg",
    "usgs_forest_neg2.jpg": "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/43/USGS_Team_Flies_Drones_for_Oregon_BLM_Forestry_Research_%2814757978546%29.jpg/1280px-USGS_Team_Flies_Drones_for_Oregon_BLM_Forestry_Research_%2814757978546%29.jpg",
}

MISS = ROOT / "data" / "afse_test_ours" / "images" / (
    "Original-Sized-RGB-Frame-45854-_jpg.rf.671ef670a6fcc276fb1714e7af2d3a2e.jpg"
)


def dl(name, url):
    p = OUT / name
    if not p.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "samolet-probe/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
            f.write(r.read())
    im = Image.open(p).convert("RGB")
    print(f"dl {name}: {im.size}", flush=True)
    return p


def show_dets(tag, res, conf):
    dets = []
    if res.boxes is not None and len(res.boxes):
        for b in res.boxes:
            dets.append((int(b.cls), round(float(b.conf), 3)))
    dets.sort(key=lambda x: -x[1])
    names = {0: "smoke", 1: "fire"}
    s = ", ".join(f"{names[c]}:{v}" for c, v in dets[:8]) or "quiet"
    flag = "ALARM" if any(c == 1 and v >= conf for c, v in dets) else "----"
    print(f"[{flag}] {tag} conf={conf}: {s}", flush=True)
    return dets


def main():
    m = YOLO(str(ROOT / "runs" / "mixed_v3_final" / "weights" / "best.pt"))
    files = [dl(n, u) for n, u in URLS.items()]

    # кроп-зонды пропущенного кадра: огонь крупно + дымовое пятно отдельно
    im = Image.open(MISS).convert("RGB")
    W, H = im.size
    # GT fire box: x=0.5247 y=0.0819 w=0.0207 h=0.0338
    fx, fy = 0.5247 * W, 0.0819 * H
    crops = {
        "miss_crop_fire_4x.jpg": im.crop((fx - 160, fy - 120, fx + 160, fy + 120)),
        "miss_crop_smoke_plume.jpg": im.crop((int(0.65 * W), int(0.18 * H), W, H)),
    }
    for n, c in crops.items():
        c.save(OUT / n)
        files.append(OUT / n)
        print(f"crop {n}: {c.size}", flush=True)

    for conf in (0.1, 0.05):
        print(f"--- conf={conf} ---", flush=True)
        for p in files:
            r = m.predict(str(p), imgsz=960, conf=conf, verbose=False)[0]
            show_dets(p.name, r, conf)
            if conf == 0.05:
                r.save(filename=str(OUT / (p.stem + "_ann.jpg")))
    print("annotated ->", OUT)


if __name__ == "__main__":
    main()
