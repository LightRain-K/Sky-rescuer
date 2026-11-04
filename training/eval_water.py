"""Единый тест водной модели: SeaDronesSee val + кросс-домен + отчёт.

Запуск из корня:
    python training/eval_water.py [ВЕСА] [--conf 0.25] [--tag заметка]

По умолчанию: runs/sea_yolo11n_960/weights/best.pt, conf=0.25, imgsz=960.
Пишет reports/eval_sea_ГГГГММДД_ЧЧММСС/report.md + аннотации пропусков.
Holdout для воды = официальный val (test без разметки).
Кросс-домен (информативно, без вердикта): AFSE holdout + 150 кадров D-Fire.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]

# База первого прогона sea_yolo11n_960 (conf=0.25, imgsz=960) — conf обязан
# совпадать с тем, что передаётся в eval, иначе вердикт врёт.
BASE = {"swimmer_r": 0.803, "swimmer_map": 0.759, "all_map": 0.764}
TOL = 0.01


def parse_args(argv):
    weights = str(ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt")
    conf, tag = 0.25, ""
    rest = []
    it = iter(argv)
    for a in it:
        if a.startswith("--conf="):
            conf = float(a.split("=", 1)[1])
        elif a == "--conf":
            conf = float(next(it))
        elif a.startswith("--tag="):
            tag = a.split("=", 1)[1]
        elif a == "--tag":
            tag = next(it)
        elif a == "--help":
            print(__doc__)
            sys.exit(0)
        else:
            rest.append(a)
    if rest:
        weights = rest[0]
    return weights, conf, tag


def alarm_loop(model, img_paths, lab_of, conf, out_dir, save_miss_cls=0, save_n=6):
    """Тревога на кадре по каждому классу. lab_of: stem -> set(cls)."""
    hit = {c: 0 for c in range(5)}
    n = {c: 0 for c in range(5)}
    misses, saved = [], 0
    for ip in img_paths:
        cls = lab_of(ip.stem)
        r = model.predict(str(ip), imgsz=960, conf=conf, verbose=False)[0]
        best = {}
        for b in (r.boxes or []):
            c, v = int(b.cls), float(b.conf)
            best[c] = max(best.get(c, 0.0), v)
        for c in cls:
            n[c] += 1
            hit[c] += best.get(c, 0.0) > 0
        if save_miss_cls in cls and best.get(save_miss_cls, 0.0) == 0 and saved < save_n:
            r.save(filename=str(out_dir / f"miss_{ip.stem[:50]}.jpg"))
            saved += 1
            misses.append(ip.stem)
    return hit, n, misses


def main():
    from ultralytics import YOLO

    weights, conf, tag = parse_args(sys.argv[1:])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = ROOT / "reports" / f"eval_sea_{stamp}"
    rep.mkdir(parents=True, exist_ok=True)
    print(f"weights: {weights} | conf={conf}", flush=True)

    t0 = time.time()
    model = YOLO(weights)
    r = model.val(data=str(ROOT / "data" / "seax.yaml"), split="val",
                  imgsz=960, conf=conf, verbose=False, plots=False, batch=1)
    m = {NAMES[i]: (float(r.box.p[i]), float(r.box.r[i]), float(r.box.ap50[i])) for i in range(5)}
    for c in NAMES:
        print(f"{c}: P={m[c][0]:.3f} R={m[c][1]:.3f} mAP50={m[c][2]:.3f}", flush=True)

    d = ROOT / "data" / "seadronessee" / "data"
    imgs = sorted((d / "images" / "val").rglob("*.jpg"))

    def lab_of(stem):
        lp = d / "labels" / "val" / "shard_000" / (stem + ".txt")
        if not lp.exists():
            return set()
        return {int(ln.split()[0]) for ln in lp.read_text().splitlines()
                if len(ln.split()) >= 5}

    hit, n, misses = alarm_loop(model, imgs, lab_of, conf, rep)
    alarm = {NAMES[c]: f"{hit[c]}/{n[c]}" for c in range(5)}
    print("alarm:", alarm, flush=True)

    # кросс-домен: что водная модель видит на суше (должна молчать)
    cross = {}
    afse = sorted((ROOT / "data" / "afse_test_ours" / "images").glob("*.jpg"))
    dfire = sorted((ROOT / "data" / "dfire" / "test" / "images").glob("*.jpg"))[:150]
    for name, paths in (("afse_holdout", afse), ("dfire150", dfire)):
        det = 0
        for ip in paths:
            rr = model.predict(str(ip), imgsz=960, conf=conf, verbose=False)[0]
            det += len(rr.boxes or [])
        cross[name] = f"{det} боксов / {len(paths)} кадров"
        print(f"cross {name}: {cross[name]}", flush=True)

    ok = (m["swimmer"][1] >= BASE["swimmer_r"] - TOL
          and m["swimmer"][2] >= BASE["swimmer_map"] - TOL)
    verdict = "PASS" if ok else "CHECK"
    (rep / "report.md").write_text(
        f"# Eval SEA {stamp} {verdict}\n\n"
        f"weights: `{weights}` | conf={conf} imgsz=960"
        f"{(' | ' + tag) if tag else ''} | {time.time() - t0:.0f}s\n\n"
        f"## SeaDronesSee val\n\n| | P | R | mAP50 | alarm |\n|---|---|---|---|---|\n"
        + "".join(f"| {c} | {m[c][0]:.3f} | {m[c][1]:.3f} | {m[c][2]:.3f} | {alarm[c]} |\n" for c in NAMES)
        + f"\nПропуски swimmer ({len(misses)}): " + (", ".join(f"`{s[:45]}`" for s in misses) or "нет")
        + f"\n\n## Кросс-домен (водная на суше)\n\n"
        + "".join(f"- {k}: {v}\n" for k, v in cross.items()),
        encoding="utf-8",
    )
    print(f"verdict: {verdict} -> {rep / 'report.md'}")


if __name__ == "__main__":
    main()
