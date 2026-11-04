"""Прототип трекинга воды: детект -> ByteTrack -> телеметрия -> скоринг v0.

Логика оператора, закодированная в v0 (пороги — первые прикидки,
калибруются на своих полётах):
  swimmer-трек с возрастом >= 20 кадров получает баллы:
    +1 дрейф: средняя скорость < 2 px/кадр (не гребёт к безопасности)
    +1 изоляция: мин. дистанция до любой лодки > 300 px
    +1 долго в воде: возраст трека > 60 кадров
  Приоритет 0..3, оператор смотрит сверху вниз.

Запуск из корня:
    python training/track_water.py [--seq 13907..14087] [--conf 0.25]
    python training/track_water.py --video=data/video_probe/lone_swimmer.mp4

Выход: runs/tracks/<seq>/track.mp4 + telemetry.json + ранжир в консоль.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]

# Пороги v0 (пиксели при imgsz=960, кадры). Калибровать на своих полётах!
MIN_AGE, DRIFT_PX, FAR_BOAT_PX, LONG_AGE = 20, 2.0, 300.0, 60


def parse_seq(argv):
    seq, conf, video = "13907..14087", 0.25, None
    for a in argv:
        if a.startswith("--seq="):
            seq = a.split("=", 1)[1]
        elif a.startswith("--conf="):
            conf = float(a.split("=", 1)[1])
        elif a.startswith("--video="):
            video = a.split("=", 1)[1]
        elif a == "--help":
            print(__doc__)
            sys.exit(0)
    return seq, conf, video


def main():
    from ultralytics import YOLO

    seq, conf, video = parse_seq(sys.argv[1:])
    model = YOLO(str(ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt"))

    if video:
        out = ROOT / "runs" / "tracks" / Path(video).stem
        out.mkdir(parents=True, exist_ok=True)
        print(f"video {video}", flush=True)
        results = [(fi, r) for fi, r in enumerate(
            model.track(source=video, imgsz=960, conf=conf, verbose=False,
                        tracker="bytetrack.yaml", stream=True))]
    else:
        s, e = (int(x) for x in seq.split(".."))
        imgdir = ROOT / "data" / "seadronessee" / "data" / "images" / "val" / "shard_000"
        paths = [imgdir / f"{i}.jpg" for i in range(s, e + 1) if (imgdir / f"{i}.jpg").exists()]
        print(f"seq {seq}: {len(paths)} кадров", flush=True)
        out = ROOT / "runs" / "tracks" / f"seq{seq.replace('.', '_')}"
        out.mkdir(parents=True, exist_ok=True)
        results = []
        for p in paths:
            r = model.track(str(p), imgsz=960, conf=conf, verbose=False,
                            persist=True, tracker="bytetrack.yaml")[0]
            results.append((0, r))

    tracks = defaultdict(list)  # id -> [(cx, cy, cls)]
    boats = defaultdict(list)   # frame_idx -> [(cx, cy)]
    writer, size = None, None
    for fi, r in results:
        ann = r.plot()
        if writer is None:
            h, w = ann.shape[:2]
            size = (w, h)
            writer = cv2.VideoWriter(str(out / "track.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
        writer.write(ann)
        if r.boxes is not None and r.boxes.id is not None:
            for b, tid in zip(r.boxes, r.boxes.id.tolist()):
                x1, y1, x2, y2 = b.xyxy.tolist()[0]
                cx, cy, c = (x1 + x2) / 2, (y1 + y2) / 2, int(b.cls)
                tracks[int(tid)].append((cx, cy, c, fi))
                if c == 1:
                    boats[fi].append((cx, cy))
    writer.release()

    # телеметрия + скоринг
    report = []
    for tid, pts in sorted(tracks.items()):
        cls = max({p[2] for p in pts}, key=lambda c: sum(1 for p in pts if p[2] == c))
        age = len(pts)
        disp = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
        speed = disp / max(age - 1, 1)
        far = -1.0
        if cls == 0:
            ds = [((x - bx) ** 2 + (y - by) ** 2) ** 0.5
                  for (x, y, c, fi) in pts for (bx, by) in boats.get(fi, [])]
            far = min(ds) if ds else float("inf")
        score, why = 0, []
        if cls == 0 and age >= MIN_AGE:
            if speed < DRIFT_PX:
                score += 1
                why.append(f"дрейф {speed:.1f}px/к")
            if far > FAR_BOAT_PX:
                score += 1
                why.append(f"до лодки {far:.0f}px" if far != float("inf") else "лодок нет")
            if age >= LONG_AGE:
                score += 1
                why.append(f"в воде {age}к")
        report.append({"id": tid, "cls": NAMES[cls], "age": age,
                       "disp": round(disp, 1), "speed": round(speed, 2),
                       "far_boat": None if far < 0 else (round(far, 1) if far != float("inf") else "none"),
                       "score": score, "why": why})
    (out / "telemetry.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    sw = sorted([t for t in report if t["cls"] == "swimmer"],
                key=lambda t: (-t["score"], -t["age"]))
    print(f"треков: {len(report)}, пловцов: {len(sw)}", flush=True)
    for t in sw[:10]:
        print(f"  id={t['id']} score={t['score']} age={t['age']} "
              f"disp={t['disp']} {'| '.join(t['why'])}", flush=True)
    print("видео:", out / "track.mp4")


if __name__ == "__main__":
    main()
