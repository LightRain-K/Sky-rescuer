"""Бортовой конвейер: камера -> детекция -> подтверждение 3 из 5 -> тревога.

Один скрипт на оба профиля миссии. Работает и на ноутбуке (демо, отладка),
и на борту (Raspberry Pi 5, NCNN-модель).

Примеры:
    # вода, файл, без окна
    python onboard/pipeline.py --profile water --source data/video_probe/beach_boats.mp4 --no-display

    # пожар, веб-камера, с окном
    python onboard/pipeline.py --profile fire --source 0

    # папка кадров (D-Fire test) + фиксированные координаты для пересчёта
    python onboard/pipeline.py --profile fire --source data/dfire/test/images \
        --gps 55.16,61.40,120,90 --max-frames 200

    # с телеметрией по MAVLink
    python onboard/pipeline.py --profile water --source 0 --mavlink udp:127.0.0.1:14550

Выход: runs/onboard/<profile>_<время>/
    alarms.jsonl   — тревоги: класс, уверенность, bbox, координаты
    annotated.mp4  — видео с разметкой (подтверждённые тревоги — жирным)
    summary.txt    — сколько кадров, тревог, скорость

Логика тревоги: объект должен быть найден в 3 кадрах из 5 подряд.
После тревоги по тому же объекту — пауза (cooldown), чтобы не спамить.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

PROFILES = {
    "fire": {
        "weights": ROOT / "runs" / "mixed_v3_final" / "weights" / "best.pt",
        "names": ["smoke", "fire"],
        "conf": 0.10,
        "alarm_classes": [1],          # тревога только по огню, дым — ранний признак
        "colors": [(200, 180, 120), (40, 40, 220)],
    },
    "water": {
        "weights": ROOT / "runs" / "sea_yolo11n_960" / "weights" / "best.pt",
        "names": ["swimmer", "boat", "jetski", "life_saving", "buoy"],
        "conf": 0.25,
        "alarm_classes": [0],          # тревога по пловцу
        "colors": [(220, 120, 40), (200, 200, 40), (40, 160, 220), (60, 200, 60), (220, 60, 200)],
    },
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES), default="water")
    ap.add_argument("--source", default="0", help="индекс камеры, файл или папка с кадрами")
    ap.add_argument("--weights", default=None, help="переопределить веса (в т.ч. ncnn-папка)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="0 — без ограничения")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="не писать annotated.mp4")
    ap.add_argument("--gps", default=None, help="lat,lon,alt_m,heading_deg — фиксированные для демо")
    ap.add_argument("--mavlink", default=None, help="строка подключения, напр. udp:127.0.0.1:14550")
    ap.add_argument("--hfov", type=float, default=66.0, help="горизонтальный угол обзора камеры, град")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--need", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=60)
    return ap.parse_args(argv)


# ---------------------------------------------------------------- источники
def frame_source(src: str):
    """Возвращает (генератор кадров, описание, fps_подсказка)."""
    p = Path(src)
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        def gen():
            for f in files:
                im = cv2.imread(str(f))
                if im is not None:
                    yield im
        return gen(), f"папка {p.name} ({len(files)} кадров)", 10.0

    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise SystemExit(f"камера {src} не открылась")
        def gen():
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                yield im
        return gen(), f"камера {src}", cap.get(cv2.CAP_PROP_FPS) or 30.0

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"не открылось: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    def gen():
        while True:
            ok, im = cap.read()
            if not ok:
                break
            yield im
    return gen(), f"видео {p.name}", fps


# ---------------------------------------------------------------- геометрия
class GeoProjector:
    """Пиксель -> координаты на земле. Допущение: камера смотрит вниз (nadir),
    подстилающая поверхность плоская. Для наклонной камеры нужна матрица
    поворота из телеметрии — это следующий шаг."""

    def __init__(self, lat, lon, alt, heading, hfov):
        self.lat, self.lon, self.alt = lat, lon, alt
        self.heading = math.radians(heading)
        self.hfov = math.radians(hfov)

    def project(self, cx, cy, w, h):
        if self.alt is None:
            return None, None, None
        vfov = 2 * math.atan(math.tan(self.hfov / 2) * h / w)
        ax = math.atan((cx - w / 2) / (w / 2) * math.tan(self.hfov / 2))
        ay = math.atan((cy - h / 2) / (h / 2) * math.tan(vfov / 2))
        dx = self.alt * math.tan(ax)
        dy = self.alt * math.tan(ay)
        # поворот на курс: dx — вправо, dy — вперёд
        east = dx * math.cos(self.heading) + dy * math.sin(self.heading)
        north = -dx * math.sin(self.heading) + dy * math.cos(self.heading)
        lat = self.lat + north / 111_320.0
        lon = self.lon + east / (111_320.0 * max(math.cos(math.radians(self.lat)), 1e-6))
        return round(lat, 7), round(lon, 7), round(math.hypot(dx, dy), 1)


class MavlinkGPS:
    """Читает GLOBAL_POSITION_INT без блокировки. Если связи нет — молча None."""

    def __init__(self, conn):
        from pymavlink import mavutil
        self.m = mavutil.mavlink_connection(conn)
        self.state = None

    def read(self):
        try:
            msg = self.m.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            if msg:
                self.state = (msg.lat / 1e7, msg.lon / 1e7,
                              msg.relative_alt / 1000.0, msg.hdg / 100.0)
        except Exception:
            pass
        return self.state


# ---------------------------------------------------------------- тревоги
class AlarmGate:
    """3 из 5 кадров по каждому объекту, затем пауза."""

    def __init__(self, window=5, need=3, cooldown=90):
        self.window, self.need, self.cooldown = window, need, cooldown
        self.hist = defaultdict(lambda: deque(maxlen=window))
        self.last_alarm = {}
        self.recent = []          # (frame, gx, gy, cls) — горячие ячейки кадра

    def update(self, tid, hit, frame_idx, cls, centre=None, frame_wh=None):
        key = (tid, cls)
        self.hist[key].append(bool(hit))
        if not hit:
            return False
        if sum(self.hist[key]) < self.need:
            return False
        if frame_idx - self.last_alarm.get(key, -10**9) < self.cooldown:
            return False
        # одна тревога на «горячую ячейку»: при разрыве трека и при
        # множестве боксов на одном очаге оператор получает одну тревогу
        if centre is not None and frame_wh is not None:
            w, h = frame_wh
            cs = max(80, int(w / 10))
            gx, gy = int(centre[0] // cs), int(centre[1] // cs)
            for f, ax, ay, c in self.recent:
                if c == cls and frame_idx - f < self.cooldown and \
                        abs(ax - gx) <= 1 and abs(ay - gy) <= 1:
                    return False
            self.recent.append((frame_idx, gx, gy, cls))
            self.recent = [r for r in self.recent if frame_idx - r[0] < self.cooldown * 2]
        self.last_alarm[key] = frame_idx
        return True

    def confirmed(self, tid, cls):
        return sum(self.hist.get((tid, cls), [])) >= self.need


class PseudoTracker:
    """Резервный трекер на случай, если бэкенд не отдаёт id (например, часть
    экспортированных форматов): ближайший центр в пределах порога."""

    def __init__(self, max_dist=80, max_age=30):
        self.tracks = {}
        self.next_id = 1
        self.max_dist, self.max_age = max_dist, max_age

    def assign(self, centres, frame_idx):
        used = set()
        out = []
        for cx, cy in centres:
            best, best_d = None, self.max_dist
            for tid, (px, py, age) in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - px, cy - py)
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.tracks[best] = (cx, cy, frame_idx)
            out.append(best)
        for tid in list(self.tracks):
            if tid not in used and frame_idx - self.tracks[tid][2] > self.max_age:
                del self.tracks[tid]
        return out


# ---------------------------------------------------------------- основной цикл
def main(argv=None):
    args = parse_args(argv)
    prof = PROFILES[args.profile]
    weights = Path(args.weights) if args.weights else prof["weights"]
    if not weights.exists():
        raise SystemExit(f"нет весов: {weights}\nСкачай из GitHub Release или укажи --weights")
    conf = args.conf if args.conf is not None else prof["conf"]

    from ultralytics import YOLO

    model = YOLO(str(weights))

    gps = None
    if args.mavlink:
        gps = MavlinkGPS(args.mavlink)
    elif args.gps:
        fixed = tuple(float(x) for x in args.gps.split(","))
        gps = type("Fixed", (), {"read": lambda self, _f=fixed: _f})()

    stream, desc, src_fps = frame_source(args.source)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs" / "onboard" / f"{args.profile}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    alarms_path = out / "alarms.jsonl"
    alarms_path.write_text("", encoding="utf-8")

    gate = AlarmGate(args.window, args.need, args.cooldown)
    pseudo = PseudoTracker()
    writer = None
    n_frames = n_alarms = 0
    t_start = time.time()
    by_class = defaultdict(int)

    print(f"[{args.profile}] {desc} | conf={conf} imgsz={args.imgsz} | -> {out}", flush=True)

    for fi, frame in enumerate(stream):
        if args.max_frames and fi >= args.max_frames:
            break
        res = model.track(frame, imgsz=args.imgsz, conf=conf, verbose=False,
                          persist=True, tracker="bytetrack.yaml")[0]

        boxes = res.boxes
        ids = boxes.id.tolist() if boxes is not None and boxes.id is not None else None
        rows = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.tolist()
            clss = [int(c) for c in boxes.cls.tolist()]
            confs = [float(c) for c in boxes.conf.tolist()]
            if ids is None:
                centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in xyxy]
                ids = pseudo.assign(centres, fi)
            for (x1, y1, x2, y2), cls, cf, tid in zip(xyxy, clss, confs, ids):
                tid = int(tid)
                is_alarm_cls = cls in prof["alarm_classes"]
                centre = ((x1 + x2) / 2, (y1 + y2) / 2)
                fh, fw = frame.shape[:2]
                fired = (gate.update(tid, is_alarm_cls, fi, cls, centre, (fw, fh))
                         if is_alarm_cls else False)
                confirmed = is_alarm_cls and gate.confirmed(tid, cls)
                rows.append((tid, cls, cf, (x1, y1, x2, y2), confirmed, fired))
                if fired:
                    n_alarms += 1
                    by_class[prof["names"][cls]] += 1
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    h, w = frame.shape[:2]
                    lat = lon = dist = None
                    state = gps.read() if gps else None
                    if state:
                        lat, lon, dist = GeoProjector(*state, args.hfov).project(cx, cy, w, h)
                    rec = {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "frame": fi,
                        "class": prof["names"][cls],
                        "conf": round(cf, 3),
                        "track_id": tid,
                        "bbox_px": [round(v, 1) for v in (x1, y1, x2, y2)],
                        "centre_px": [round(cx, 1), round(cy, 1)],
                        "lat": lat, "lon": lon, "dist_m": dist,
                    }
                    with alarms_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"  ALARM frame={fi} {rec['class']} conf={cf:.2f} "
                          f"id={tid}" + (f" lat={lat} lon={lon}" if lat else ""), flush=True)

        n_frames += 1
        if not args.no_video or not args.no_display:
            ann = frame.copy()
            for tid, cls, cf, (x1, y1, x2, y2), confirmed, fired in rows:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                col = prof["colors"][cls % len(prof["colors"])]
                thick = 4 if confirmed else 2
                cv2.rectangle(ann, (x1, y1), (x2, y2), col, thick)
                label = f"{prof['names'][cls]} {cf:.2f} id:{tid}"
                if confirmed:
                    label = "ALARM " + label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(ann, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, y1), col, -1)
                cv2.putText(ann, label, (x1 + 2, max(y1 - 6, th)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(ann, f"{args.profile} | alarms: {n_alarms} | frame {fi}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if not args.no_video:
            if writer is None:
                h, w = ann.shape[:2]
                writer = cv2.VideoWriter(str(out / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(ann)
        if not args.no_display:
            cv2.imshow("samolet", ann)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    ms = 1000 * elapsed / max(n_frames, 1)
    summary = (
        f"profile: {args.profile}\n"
        f"source: {desc}\n"
        f"weights: {weights}\n"
        f"conf: {conf}, imgsz: {args.imgsz}\n"
        f"frames: {n_frames}\n"
        f"alarms: {n_alarms} {dict(by_class)}\n"
        f"time: {elapsed:.1f} s ({ms:.1f} ms/frame, {1000 / ms:.1f} fps)\n"
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print("alarms ->", alarms_path)


if __name__ == "__main__":
    main()
