"""Обучение merged v3: старт с newest весов (mixed_v2 -> mixed_dfire_custom).

Особенности:
- box=5.0 (ниже дефолтных 7.5): web/AFSE-боксы рыхлые и субъективные,
  не наказываем модель строго за границы — главное найти, а не обвести.
- «Глаза»: каждые 3 эпохи предсказания на 4 фиксированных сложных кадрах
  складываются в runs/mixed_v3/eyeball/e{N}/ — смотришь глазами прямо
  во время обучения. Плюс val_batch0_pred.jpg в корне runs/mixed_v3
  обновляется каждую эпоху.
- Контроль в конце: D-Fire test + AFSE holdout (тревоги).

Запуск:
    python training/train_mixed_v3.py
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mixed_v2 import presence_rate

ROOT = Path(__file__).resolve().parent.parent
EYE = ROOT / "runs" / "mixed_v3" / "eyeball"

PROBES = [
    ROOT / "data" / "web_work" / "images" / "DJI_0141.jpg",
    ROOT / "data" / "web_work" / "images" / "1.jpg",
    ROOT / "data" / "afse_test_ours" / "images" / (
        "Drone-Footage-Fire-Maui-Clint_Hansen_1130pm_1280x720-30-"
        "mp4_frame1590_jpg.rf.795c947e79b2fea0d159c7c327578176.jpg"),
    ROOT / "data" / "web_work" / "images" / (
        "v-voskresene-v-chuvashii-zaregistrirovany-tri-ochaga-lesnyh-pozharov_"
        "17204245691627092388__2000x2000__watermark.jpg"),
]


def probe_with(weights: str, tag: str, device=None):
    probes = [str(p) for p in PROBES if p.exists()]
    print(f"eyeball [{tag}]: {len(probes)} probes")
    YOLO(weights).predict(probes, imgsz=960, conf=0.1, verbose=False,
                          save=True, project=str(EYE), name=tag, device=device)


def newest_weights() -> Path:
    for cand in ("runs/mixed_v3_final/weights/best.pt",
                 "runs/mixed_v2/weights/best.pt",
                 "runs/mixed_dfire_custom/weights/best.pt",
                 "runs/dfire_yolo11s_960/weights/best.pt"):
        if (ROOT / cand).exists():
            return ROOT / cand
    raise FileNotFoundError("нет стартовых весов")


def _eyeball(trainer):
    # Срабатывает на on_fit_epoch_end у трейнера (у него есть .epoch).
    # last.pt может отставать на эпоху — для глазок не важно.
    ep = getattr(trainer, "epoch", 0)
    if (ep + 1) % 3 != 0:
        return
    src = Path(trainer.save_dir) / "weights" / "last.pt"
    if not src.exists():
        return
    EYE.mkdir(parents=True, exist_ok=True)
    snap = EYE / f"_snap_e{ep + 1}.pt"
    shutil.copy(src, snap)
    try:
        probe_with(str(snap), f"e{ep + 1}", device="cpu")
    finally:
        snap.unlink(missing_ok=True)


def main():
    w = newest_weights()
    print("старт с:", w)
    EYE.mkdir(parents=True, exist_ok=True)
    probe_with(str(w), "e0_start")  # было ДО обучения — точка отсчёта

    model = YOLO(str(w))
    model.add_callback("on_fit_epoch_end", _eyeball)
    model.train(
        data=str(ROOT / "data" / "merged_v3.yaml"),
        epochs=15,
        imgsz=960,python tune_gates.py night.jpg --t0 2.2 --t1 6.0 --sigma 24 --out tune_out
        batch=8,
        device=0,
        workers=4,
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=2,
        patience=6,
        box=6.0,
        plots=True,
        project=str(ROOT / "runs"),
        name="mixed_v3",
    )
    print("--- D-Fire test (регрессия) ---")
    m = model.val(data=str(ROOT / "data" / "dfire.yaml"), split="test",
                  imgsz=960, conf=0.1, verbose=False)
    print(f"fire P={m.box.p[1]:.3f} R={m.box.r[1]:.3f} mAP50={m.box.ap50[1]:.3f} | "
          f"smoke R={m.box.r[0]:.3f} mAP50={m.box.ap50[0]:.3f}")
    print("--- AFSE holdout (тревоги на кадрах с огнём) ---")
    hit, tot = presence_rate(model, ROOT / "data" / "afse_test_ours" / "images",
                             ROOT / "data" / "afse_test_ours" / "labels")
    print(f"fire images flagged {hit}/{tot} = {hit / tot:.3f} (было 25/29=0.862)")


if __name__ == "__main__":
    main()
