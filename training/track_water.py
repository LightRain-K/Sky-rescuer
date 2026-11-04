"""Прототип трекинга воды: детект -> ByteTrack -> телеметрия -> скоринг v0.

Класс трека — взвешенное голосование уверенностью по всей истории
(убирает мерцание swimmer/boat/jetski на мелких целях); стабильность
видна в поле stab телеметрии. Оверлей: свой, с подложками подписей.

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

    tracks = defaultdict(list)  # id -> [(cx, cy, cls, conf, fi, x1,y1,x2,y2)]
    for fi, r in results:
        if r.boxes is not None and r.boxes.id is not None:
            for b, tid in zip(r.boxes, r.boxes.id.tolist()):
                x1, y1, x2, y2 = b.xyxy.tolist()[0]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tracks[int(tid)].append((cx, cy, int(b.cls), float(b.conf), fi,
                                         x1, y1, x2, y2))

    # сглаженный класс трека: взвешенное голосование уверенностью.
    # Убирает мерцание swimmer/boat/jetski на мелких целях.
    def smooth(pts):
        votes = defaultdict(float)
        for p in pts:
            votes[p[2]] += p[3]
        return max(votes, key=votes.get)

    scls = {tid: smooth(pts) for tid, pts in tracks.items()}
    agree = {tid: sum(1 for p in pts if p[2] == scls[tid]) / len(pts)
             for tid, pts in tracks.items()}
    boats = defaultdict(list)  # fi -> [(cx, cy)] только стабильных лодок
    for tid, pts in tracks.items():
        if scls[tid] == 1:
            for (cx, cy, c, cf, fi, *_b) in pts:
                boats[fi].append((cx, cy))

    # оверлей со сглаженными классами
    COLORS = [(255, 0, 0), (0, 255, 255), (0, 165, 255), (0, 255, 0), (255, 0, 255)]
    writer = None
    for fi, r in results:
        ann = r.orig_img.copy()
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for b, tid in zip(r.boxes, ids):
                x1, y1, x2, y2 = (int(v) for v in b.xyxy.tolist()[0])
                if tid is None:  # детект без трека: показываем как есть
                    c, tag = int(b.cls), "new"
                else:
                    tid, c, tag = int(tid), scls[int(tid)], f"id:{int(tid)}"
                cv2.rectangle(ann, (x1, y1), (x2, y2), COLORS[c], 2)
                tag = f"{tag} {NAMES[c]}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                inside = (y2 - y1) > 40  # подпись внутрь высоких боксов, чтобы не лепилась на соседей
                ty = y1 + th + 8 if inside else max(y1 - 6, th)
                cv2.rectangle(ann, (x1, max(ty - th - 8, 0)), (x1 + tw + 4, ty),
                              COLORS[c], -1)
                cv2.putText(ann, tag, (x1 + 2, max(ty - 6, th)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if writer is None:
            h, w = ann.shape[:2]
            writer = cv2.VideoWriter(str(out / "track.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(ann)
    writer.release()

    # телеметрия + скоринг (класс — сглаженный по треку)
    report = []
    for tid, pts in sorted(tracks.items()):
        cls = scls[tid]
        age = len(pts)
        disp = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
        speed = disp / max(age - 1, 1)
        far = -1.0
        if cls == 0:
            ds = [((x - bx) ** 2 + (y - by) ** 2) ** 0.5
                  for (x, y, c, cf, fi, *_b) in pts for (bx, by) in boats.get(fi, [])]
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
                       "stab": round(agree[tid], 2),
                       "disp": round(disp, 1), "speed": round(speed, 2),
                       "far_boat": None if far < 0 else (round(far, 1) if far != float("inf") else "none"),
                       "score": score, "why": why})
    (out / "telemetry.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    sw = sorted([t for t in report if t["cls"] == "swimmer"],
                key=lambda t: (-t["score"], -t["age"]))
    print(f"треков: {len(report)}, пловцов: {len(sw)}", flush=True)
    for t in sw[:10]:
        print(f"  id={t['id']} score={t['score']} age={t['age']} stab={t['stab']} "
              f"disp={t['disp']} {'| '.join(t['why'])}", flush=True)
    print("видео:", out / "track.mp4")


if __name__ == "__main__":
    main()
