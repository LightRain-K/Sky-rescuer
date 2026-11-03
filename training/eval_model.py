"""Единая батарея тестов модели: D-Fire test + AFSE holdout + отчёт.

Запуск из корня проекта:
    python training/eval_model.py [ВЕСА] [--conf 0.1] [--tag заметка]

По умолчанию: runs/mixed_v3_final/weights/best.pt, conf=0.1, imgsz=960.
D-Fire test всегда с batch=1 (батчевый val виснет на этом сплите).
Пишет reports/eval_ГГГГММДД_ЧЧММСС.md + аннотации пропусков holdout.

Правило приёмки: fire-recall D-Fire не хуже 0.655 и holdout не хуже 28/29.
AFSE holdout (data/afse_test_ours) НИКОГДА не используется в обучении.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Базовые значения финальной модели (conf=0.1, imgsz=960, округлены).
# Сравнение с допуском TOL, чтобы шум округления не давал ложный CHECK.
TOL = 0.003
BASE = {
    "fire_p": 0.715, "fire_r": 0.655, "fire_map": 0.670,
    "smoke_p": 0.831, "smoke_r": 0.772, "smoke_map": 0.797,
    "holdout": 28 / 29,
}

FIRE, SMOKE = 1, 0  # наши классы


def parse_args(argv):
    weights = str(ROOT / "runs" / "mixed_v3_final" / "weights" / "best.pt")
    conf = 0.1
    tag = ""
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


def dfire_test(model, conf):
    r = model.val(
        data=str(ROOT / "data" / "dfire.yaml"), split="test",
        imgsz=960, conf=conf, verbose=False, plots=False, batch=1,
    )
    return {
        "fire_p": float(r.box.p[FIRE]), "fire_r": float(r.box.r[FIRE]),
        "fire_map": float(r.box.ap50[FIRE]),
        "smoke_p": float(r.box.p[SMOKE]), "smoke_r": float(r.box.r[SMOKE]),
        "smoke_map": float(r.box.ap50[SMOKE]),
    }


def holdout(model, conf, out_dir):
    """Оперативная метрика: тревога на кадре. Возвращает детали по группам."""
    d = ROOT / "data" / "afse_test_ours"
    res = {"fire_hit": 0, "fire_n": 0, "smoke_hit": 0, "smoke_n": 0,
           "empty_quiet": 0, "empty_n": 0, "misses": []}
    for t in sorted((d / "labels").glob("*.txt")):
        lines = [ln.split() for ln in t.read_text().splitlines() if ln.strip()]
        cls = {int(p[0]) for p in lines} if lines else set()
        r = model.predict(str(d / "images" / (t.stem + ".jpg")),
                          imgsz=960, conf=conf, verbose=False)[0]
        dets = [(int(b.cls), float(b.conf)) for b in (r.boxes or [])]
        fmax = max([v for c, v in dets if c == FIRE], default=0.0)
        smax = max([v for c, v in dets if c == SMOKE], default=0.0)
        if FIRE in cls:
            res["fire_n"] += 1
            if fmax > 0:
                res["fire_hit"] += 1
            else:
                res["misses"].append((t.stem, smax))
                r.save(filename=str(out_dir / f"miss_{t.stem[:60]}.jpg"))
        elif SMOKE in cls:
            res["smoke_n"] += 1
            res["smoke_hit"] += smax > 0
        else:
            res["empty_n"] += 1
            res["empty_quiet"] += (fmax == 0 and smax == 0)
    return res


def main():
    from ultralytics import YOLO

    weights, conf, tag = parse_args(sys.argv[1:])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep_dir = ROOT / "reports" / f"eval_{stamp}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    print(f"weights: {weights} | conf={conf} {('tag=' + tag) if tag else ''}", flush=True)

    t0 = time.time()
    model = YOLO(weights)
    m = dfire_test(model, conf)
    print(f"D-Fire test: fire P={m['fire_p']:.3f} R={m['fire_r']:.3f} mAP50={m['fire_map']:.3f} | "
          f"smoke P={m['smoke_p']:.3f} R={m['smoke_r']:.3f} mAP50={m['smoke_map']:.3f}", flush=True)
    h = holdout(model, conf, rep_dir)
    h05 = holdout(model, 0.05, rep_dir) if abs(conf - 0.05) > 1e-9 else None
    print(f"holdout@{conf}: fire {h['fire_hit']}/{h['fire_n']} | "
          f"smoke {h['smoke_hit']}/{h['smoke_n']} | quiet {h['empty_quiet']}/{h['empty_n']}", flush=True)
    if h05:
        print(f"holdout@0.05: fire {h05['fire_hit']}/{h05['fire_n']} | "
              f"smoke {h05['smoke_hit']}/{h05['smoke_n']} | quiet {h05['empty_quiet']}/{h05['empty_n']}", flush=True)

    ok_fire = m["fire_r"] >= BASE["fire_r"] - TOL
    ok_hold = (h["fire_hit"] / max(h["fire_n"], 1)) >= BASE["holdout"] - TOL
    verdict = "PASS" if (ok_fire and ok_hold) else "CHECK"
    md = rep_dir / "report.md"
    md.write_text(
        f"# Eval {stamp} {verdict}\n\n"
        f"weights: `{weights}` | conf={conf} imgsz=960"
        f"{(' | ' + tag) if tag else ''} | {time.time() - t0:.0f}s\n\n"
        f"## D-Fire test (batch=1)\n\n"
        f"| | P | R | mAP50 | base R/mAP |\n|---|---|---|---|---|\n"
        f"| fire | {m['fire_p']:.3f} | {m['fire_r']:.3f} | {m['fire_map']:.3f} "
        f"| {BASE['fire_r']:.3f}/{BASE['fire_map']:.3f} |\n"
        f"| smoke | {m['smoke_p']:.3f} | {m['smoke_r']:.3f} | {m['smoke_map']:.3f} "
        f"| {BASE['smoke_r']:.3f}/{BASE['smoke_map']:.3f} |\n\n"
        f"## AFSE holdout (alarm per frame)\n\n"
        f"| conf | fire | smoke-only | quiet |\n|---|---|---|---|\n"
        f"| {conf} | {h['fire_hit']}/{h['fire_n']} | {h['smoke_hit']}/{h['smoke_n']} "
        f"| {h['empty_quiet']}/{h['empty_n']} |\n"
        + (f"| 0.05 | {h05['fire_hit']}/{h05['fire_n']} | {h05['smoke_hit']}/{h05['smoke_n']} "
           f"| {h05['empty_quiet']}/{h05['empty_n']} |\n" if h05 else "")
        + f"\nПропуски огня ({len(h['misses'])}): "
        + (", ".join(f"`{s[:50]}` (smoke {v:.2f})" for s, v in h["misses"]) or "нет")
        + f"\n\nАннотации пропусков лежат рядом с отчётом.\n",
        encoding="utf-8",
    )
    print(f"verdict: {verdict} -> {md}")


if __name__ == "__main__":
    main()
