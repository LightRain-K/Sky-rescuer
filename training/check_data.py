"""Проверка раскладки данных на новой машине (без обучения и скачивания).

Запуск из корня:  python training/check_data.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEED = [
    "data/dfire/train/images", "data/dfire/val/images", "data/dfire/test/images",
    "data/dfire/train/labels",
    "data/afse/train/images", "data/afse/valid/images",
    "data/afse_test_ours/images", "data/afse_test_ours/labels",
    "data/custom/train/images",
    "data/web_work/images",
    "data/merged_v3/custom_up/images", "data/merged_v3/afse/images",
    "data/merged_v3/web/images",
]


def main():
    ok = True
    for rel in NEED:
        p = ROOT / rel
        ext = "*.txt" if p.name == "labels" else "*.jpg"
        n = len(list(p.glob(ext))) if p.exists() else 0
        mark = "OK " if p.exists() and n else "MISS"
        ok &= p.exists() and n > 0
        print(f"[{mark}] {rel} ({n} jpg)")
    print("DATA OK" if ok else "Чего-то не хватает — см. DATA.md")


if __name__ == "__main__":
    main()
