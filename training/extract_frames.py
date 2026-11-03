"""Нарезка видео на кадры для разметки.

Берёт каждый N-й кадр (по целевой частоте fps), выкидывает смазанные
(низкая дисперсия лапласиана) чтобы не размечать мусор.

Запуск:
    python training/extract_frames.py video.mp4 --out data/custom_raw --fps 2
    python training/extract_frames.py videos/ --out data/custom_raw --fps 2
"""
import argparse
from pathlib import Path

import cv2


def extract(video: Path, out: Path, fps: float, blur_thr: float, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or saved >= max_frames:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_thr:
                out_file = out / f"{video.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
        idx += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="видеофайл или папка с видео")
    ap.add_argument("--out", default="data/custom_raw", help="куда складывать кадры")
    ap.add_argument("--fps", type=float, default=2.0, help="сколько кадров в секунду брать")
    ap.add_argument("--blur", type=float, default=80.0, help="порог резкости (меньше = строже)")
    ap.add_argument("--max", type=int, default=2000, help="максимум кадров с одного видео")
    args = ap.parse_args()

    src = Path(args.input)
    vids = [src] if src.is_file() else sorted(
        [p for p in src.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for v in vids:
        n = extract(v, out, args.fps, args.blur, args.max)
        print(f"{v.name}: {n} кадров")
        total += n
    print(f"Всего: {total} кадров в {out}")


if __name__ == "__main__":
    main()
