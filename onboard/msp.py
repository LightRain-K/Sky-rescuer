"""Минимальный MSP-клиент для Betaflight/INAV (F405).

Без внешних зависимостей кроме pyserial. Поддержан MSP v1:
    MSP_API_VERSION (1)
    MSP_MOTOR       (110)
    MSP_SET_MOTOR   (214)

Протокол: '$M<' + size + cmd + payload + checksum,
checksum = size ^ cmd ^ payload[0] ^ ... ^ payload[n-1]

Внимание: MSP_SET_MOTOR крутит моторы. Пропеллеры снять.
"""
from __future__ import annotations

import struct
import time

MSP_API_VERSION = 1
MSP_MOTOR = 110
MSP_SET_MOTOR = 214


def _checksum(size: int, cmd: int, payload: bytes) -> int:
    c = size ^ cmd
    for b in payload:
        c ^= b
    return c & 0xFF


def build_frame(cmd: int, payload: bytes = b"") -> bytes:
    size = len(payload)
    return b"$M<" + bytes([size, cmd]) + payload + bytes([_checksum(size, cmd, payload)])


def parse_frame(buf: bytes):
    """Ищет в буфере '$M>' и возвращает (cmd, payload) или None."""
    i = buf.find(b"$M>")
    if i < 0 or len(buf) < i + 5:
        return None
    size = buf[i + 3]
    if len(buf) < i + 5 + size:
        return None
    cmd = buf[i + 4]
    payload = buf[i + 5:i + 5 + size]
    chk = buf[i + 5 + size]
    if chk != _checksum(size, cmd, payload):
        return None
    return cmd, payload


class MSP:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.timeout = timeout
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def request(self, cmd: int, payload: bytes = b"", expect: int | None = None,
                retries: int = 2):
        expect = cmd if expect is None else expect
        for _ in range(retries + 1):
            self.ser.reset_input_buffer()
            self.ser.write(build_frame(cmd, payload))
            self.ser.flush()
            buf = b""
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    buf += chunk
                    got = parse_frame(buf)
                    if got and got[0] == expect:
                        return got[1]
            buf = b""
        return None

    # --- команды ---
    def api_version(self):
        p = self.request(MSP_API_VERSION)
        if not p or len(p) < 3:
            return None
        return {"protocol": p[0], "api_major": p[1], "api_minor": p[2]}

    def motor_values(self):
        p = self.request(MSP_MOTOR)
        if not p or len(p) < 16:
            return None
        return list(struct.unpack("<8H", p[:16]))

    def set_motors(self, values):
        vals = list(values) + [1000] * (8 - len(values))
        vals = [max(1000, min(2000, int(v))) for v in vals[:8]]
        self.ser.write(build_frame(MSP_SET_MOTOR, struct.pack("<8H", *vals)))
        self.ser.flush()
        return vals


def self_test():
    """Проверка протокола без железа: сборка/разбор кадра и контрольная сумма."""
    ok = True
    payload = struct.pack("<8H", 1000, 1000, 1100, 1000, 1000, 1000, 1000, 1000)
    frame = build_frame(MSP_SET_MOTOR, payload)
    good = b"$M>" + bytes([len(payload), MSP_SET_MOTOR]) + payload + \
        bytes([_checksum(len(payload), MSP_SET_MOTOR, payload)])
    parsed = parse_frame(good)
    ok &= frame[:4] == b"$M<" + bytes([len(payload)])
    ok &= parsed == (MSP_SET_MOTOR, payload)
    broken = bytearray(good)
    broken[-1] ^= 0xFF
    ok &= parse_frame(bytes(broken)) is None
    print("self-test:", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
