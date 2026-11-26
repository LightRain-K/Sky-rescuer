"""Замер скорости инференса: одинаково работает на ноутбуке и на Raspberry Pi 5.

Запуск:
    python onboard/bench.py --model runs/sea_yolo11n_960/weights/best.pt
    python onboard/bench.py --model runs/sea_yolo11n_960/weights/best_ncnn_model --imgsz 960
    python onboard/bench.py --model ... --threads 4 --frames 100

Печатает: устройство, ms/кадр, fps, а также записывает JSON в runs/bench/.
Методика: 5 кадров прогрева (не считаются), затем N кадров; берётся среднее
и 95-й процентиль (по нему видно «дёрганье»).
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_frames(n, imgsz, source=None):
    if source:
        cap = cv2.VideoCapture(str(source))
        out = []
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
        if out:
            return out
    # синтетика: серый шум нужного размера — честно нагружает препроцессор
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".pt или папка *_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="0 — не ограничивать")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--source", default=None, help="видео вместо синтетики")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()

    if a.threads:
        cv2.setNumThreads(a.threads)
    try:
        import torch
        if a.threads:
            torch.set_num_threads(a.threads)
    except Exception:
        pass

    from ultralytics import YOLO

    model = YOLO(a.model)
    frames = make_frames(a.frames + a.warmup, a.imgsz, a.source)

    is_ncnn = Path(str(a.model)).name.endswith("_ncnn_model")
    if is_ncnn:
        device = "cpu"          # NCNN не использует CUDA
    elif a.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = a.device

    for f in frames[:a.warmup]:
        model.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=device)

    times = []
    for f in frames[a.warmup:]:
        t = time.perf_counter()
        model.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=device)
        times.append((time.perf_counter() - t) * 1000)

    if not times:
        raise SystemExit("нет кадров для замера")

    ms = statistics.mean(times)
    p95 = sorted(times)[int(len(times) * 0.95) - 1]
    res = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": str(a.model),
        "imgsz": a.imgsz,
        "device": device + (" (ncnn)" if is_ncnn else ""),
        "machine": platform.node(),
        "cpu": platform.processor(),
        "threads": a.threads or "auto",
        "frames": len(times),
        "ms_mean": round(ms, 1),
        "ms_p95": round(p95, 1),
        "fps": round(1000 / ms, 1),
    }
    out = ROOT / "runs" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"model   : {a.model}")
    print(f"device  : {device} | threads: {res['threads']} | imgsz: {a.imgsz}")
    print(f"frames  : {len(times)} (+{a.warmup} прогрев)")
    print(f"speed   : {ms:.1f} мс/кадр (p95 {p95:.1f}) = {1000 / ms:.1f} fps")
    print("json ->", path)


if __name__ == "__main__":
    main()
