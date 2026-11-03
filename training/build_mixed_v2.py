"""Сборка merged v2: D-Fire + custom (x30) + AFSE train/valid (x5, со свапом классов).

AFSE/Roboflow: 0=fire, 1=smoke -> наши 0=smoke, 1=fire (свап!).
AFSE test НЕ трогаем — вечный holdout (data/afse_test_ours).
D-Fire остаётся на месте, в yaml подключается путями (без копирования).

Выход:
  data/merged_v2/custom_up/{images,labels}  - custom train x30
  data/merged_v2/afse/{images,labels}       - AFSE train+valid, классы наши, x5
  data/merged_v2.yaml                        - train: [dfire, custom_up, afse], val: dfire/val

Запуск:
    python training/build_mixed_v2.py
"""
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "merged_v2"
CUSTOM_REPEAT = 30
AFSE_REPEAT = 5


def clean_label(src: Path, swap: bool) -> list[str] | None:
    """Прочитать txt, проверить, опционально свапнуть классы. None = битый."""
    out = []
    for ln in src.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            return None
        try:
            c = int(p[0])
            x, y, w, h = map(float, p[1:5])
        except ValueError:
            return None
        if swap:
            if c not in (0, 1):
                return None
            c = 1 - c
        elif c not in (0, 1):
            return None
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None
        out.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return out


def materialize(src_img_dir: Path, src_lab_dir: Path, dst: Path,
                prefix: str, repeat: int, swap: bool, stat: Counter) -> int:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(src_img_dir.glob("*.jpg")):
        lab = src_lab_dir / (img.stem + ".txt")
        lines = clean_label(lab, swap) if lab.exists() else []
        if lines is None:
            stat["dropped_bad_label"] += 1
            continue
        for p in lines:
            stat[f"{prefix}_boxes_cls{p[0]}"] += 1
        for i in range(repeat):
            name = f"{prefix}_{img.stem}_r{i:02d}" if repeat > 1 else f"{prefix}_{img.stem}"
            shutil.copy(img, dst / "images" / (name + ".jpg"))
            (dst / "labels" / (name + ".txt")).write_text(
                "\n".join(lines) + "\n" if lines else ""
            )
            n += 1
            if not lines:
                stat["background_images"] += 1
    return n


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    stat: Counter = Counter()

    n_custom = materialize(
        ROOT / "data" / "custom" / "train" / "images",
        ROOT / "data" / "custom" / "train" / "labels",
        OUT / "custom_up", "custom", CUSTOM_REPEAT, False, stat,
    )
    n_afse = 0
    for split in ("train", "valid"):
        n_afse += materialize(
            ROOT / "data" / "afse" / split / "images",
            ROOT / "data" / "afse" / split / "labels",
            OUT / "afse", "afse", AFSE_REPEAT, True, stat,
        )
    n_dfire = len(list((ROOT / "data" / "dfire" / "train" / "images").glob("*.jpg")))

    (ROOT / "data" / "merged_v2.yaml").write_text(
        "train:\n"
        "  - dfire/train/images\n"
        "  - merged_v2/custom_up/images\n"
        "  - merged_v2/afse/images\n"
        "val: dfire/val/images\n"
        "\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    total = n_dfire + n_custom + n_afse
    print(f"dfire train : {n_dfire} ({100 * n_dfire / total:.1f}%)")
    print(f"custom x{CUSTOM_REPEAT}: {n_custom} ({100 * n_custom / total:.1f}%)")
    print(f"afse   x{AFSE_REPEAT} : {n_afse} ({100 * n_afse / total:.1f}%)")
    print(f"total train : {total}")
    print("boxes:", dict(stat))
    print("yaml: data/merged_v2.yaml (AFSE test остался нетронутым holdout)")


if __name__ == "__main__":
    main()
