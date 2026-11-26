"""Демо моторов через F405: проверка связи и последовательность вращения.

!!! ПРОПЕЛЛЕРЫ СНЯТЬ. Моторы крутятся на столе, в защитном кожухе/клетке. !!!

По умолчанию — сухой прогон (ничего не отправляется), чтобы можно было
проверить логику без платы.

Примеры:
    python onboard/motor_demo.py --probe                       # прочитать версию API
    python onboard/motor_demo.py --port COM5 --sequence motors  # сухой прогон
    python onboard/motor_demo.py --port COM5 --sequence motors --arm
    python onboard/motor_demo.py --port /dev/ttyAMA0 --sequence alarms \
        --alarm-file runs/onboard/water_.../alarms.jsonl --arm

Что делает --sequence motors:
    по очереди крутит моторы 1..4 по 0.8 с, затем все вместе 1.5 с, затем стоп.

--sequence alarms:
    следит за файлом тревог и на каждую новую тревогу «дёргает» моторы
    (1-2, затем 3-4) — визуальная реакция борта на обнаружение.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from msp import MSP, self_test

STOP = [1000] * 8


def spin(msp, motors, throttle, seconds, dry):
    vals = STOP.copy()
    for m in motors:
        vals[m] = throttle
    if dry:
        print(f"  [dry] моторы {[m + 1 for m in motors]} -> {throttle} на {seconds} с")
        time.sleep(min(seconds, 0.15))
        return
    msp.set_motors(vals)
    time.sleep(seconds)
    msp.set_motors(STOP)


def sequence_motors(msp, throttle, dry):
    for m in range(4):
        spin(msp, [m], throttle, 0.8, dry)
        time.sleep(0.2)
    spin(msp, [0, 1, 2, 3], throttle, 1.5, dry)


def sequence_alarms(msp, alarm_file, throttle, dry, max_events=0, from_start=False):
    p = Path(alarm_file)
    if not p.exists():
        raise SystemExit(f"нет файла тревог: {p}")
    print(f"слежу за {p} ... (Ctrl+C для выхода)")
    pos = 0 if from_start else p.stat().st_size
    events = 0
    while True:
        if p.stat().st_size > pos:
            with p.open("r", encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cls = rec.get("class", "?")
                    print(f"  тревога: {cls} conf={rec.get('conf')} id={rec.get('track_id')}")
                    spin(msp, [0, 1], throttle, 0.25, dry)
                    spin(msp, [2, 3], throttle, 0.25, dry)
                    events += 1
                    if max_events and events >= max_events:
                        print("достигнут лимит событий, выхожу")
                        return
                pos = f.tell()
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="COM5 (Windows) или /dev/ttyAMA0 (Pi)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sequence", choices=["motors", "alarms"], default="motors")
    ap.add_argument("--throttle", type=int, default=1150, help="1000 = стоп, 2000 = макс")
    ap.add_argument("--alarm-file", default=None)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--from-start", action="store_true",
                    help="читать тревоги с начала файла (для проверки)")
    ap.add_argument("--arm", action="store_true", help="РЕАЛЬНО отправлять команды моторам")
    ap.add_argument("--probe", action="store_true", help="только прочитать MSP_API_VERSION")
    ap.add_argument("--self-test", action="store_true", help="проверить протокол без железа")
    a = ap.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)

    dry = not a.arm
    if dry:
        print("=" * 60)
        print("СУХОЙ ПРОГОН: команды не отправляются. Для реального — добавь --arm")
        print("=" * 60)
    else:
        print("!" * 60)
        print("ПРОПЕЛЛЕРЫ СНЯТЬ. Моторы будут крутиться физически.")
        print("!" * 60)
        time.sleep(2)

    if not a.port:
        raise SystemExit("нужен --port (COM5 / /dev/ttyAMA0)")

    if dry and not a.probe:
        msp = None
    else:
        msp = MSP(a.port, a.baud)

    try:
        if a.probe:
            print("API:", msp.api_version())
            print("моторы:", msp.motor_values())
            return

        if a.sequence == "motors":
            sequence_motors(msp, a.throttle, dry)
        else:
            if not a.alarm_file:
                raise SystemExit("для --sequence alarms нужен --alarm-file")
            sequence_alarms(msp, a.alarm_file, a.throttle, dry, a.max_events,
                            a.from_start)
    except KeyboardInterrupt:
        print("\nпрервано")
    finally:
        if msp is not None:
            try:
                msp.set_motors(STOP)
                print("моторы остановлены")
            finally:
                msp.close()


if __name__ == "__main__":
    main()
