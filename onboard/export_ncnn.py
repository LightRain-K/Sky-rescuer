"""Экспорт моделей в NCNN для борта (Raspberry Pi 5).

Запуск:
    python onboard/export_ncnn.py --profile water
    python onboard/export_ncnn.py --profile fire
    python onboard/export_ncnn.py --weights runs/sea_yolo11n_960/weights/best.pt --imgsz 960

Выход: рядом с весами появится папка <имя>_ncnn_model/ — её и копировать на борт.
NCNN не требует PyTorch: на Pi ставятся только ncnn + ultralytics (или чистый
inference-скрипт), что экономит память и ускоряет старт.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROFILES = {
    "fire": ROOT / "runs" / "mixed_v3_final" / "weights" / "best.pt",
    "water": ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES), default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--half", action="store_true", help="fp16 (только если борт поддерживает)")
    a = ap.parse_args()

    w = Path(a.weights) if a.weights else (PROFILES[a.profile] if a.profile else None)
    if w is None:
        raise SystemExit("укажи --profile или --weights")
    if not w.exists():
        raise SystemExit(f"нет весов: {w}")

    from ultralytics import YOLO

    print(f"экспорт {w.name} @ {a.imgsz} -> ncnn ...", flush=True)
    model = YOLO(str(w))
    out = model.export(format="ncnn", imgsz=a.imgsz, half=a.half)
    print("готово:", out)
    print("\nСкопируй папку на борт и проверь скорость:")
    print(f"  python onboard/bench.py --model <путь к папке ncnn> --imgsz {a.imgsz}")


if __name__ == "__main__":
    main()
