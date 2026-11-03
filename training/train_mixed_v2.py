"""Обучение merged v2: старт с mixed-весов, D-Fire + custom + AFSE.

Контроль: val по D-Fire (анти-забывание) + после обучения:
  1) D-Fire test (регрессия базы),
  2) AFSE holdout data/afse_test_ours (тревоги на кадрах с огнём),
  3) custom train (не забыли ближний вид).

Запуск:
    python training/train_mixed_v2.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def presence_rate(model, img_dir: Path, lab_dir: Path) -> tuple[int, int]:
    hit = tot = 0
    for ip in sorted(img_dir.glob("*.jpg")):
        lab = lab_dir / (ip.stem + ".txt")
        if not lab.exists():
            continue
        if not any(ln.split()[0] == "1" for ln in lab.read_text().splitlines()):
            continue
        tot += 1
        r = model.predict(str(ip), imgsz=960, conf=0.1, verbose=False)[0]
        if r.boxes is not None and any(int(c) == 1 for c in r.boxes.cls.tolist()):
            hit += 1
    return hit, tot


def main():
    model = YOLO(str(ROOT / "runs" / "mixed_dfire_custom" / "weights" / "best.pt"))
    model.train(
        data=str(ROOT / "data" / "merged_v2.yaml"),
        epochs=15,
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=2,
        patience=6,
        project=str(ROOT / "runs"),
        name="mixed_v2",
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
