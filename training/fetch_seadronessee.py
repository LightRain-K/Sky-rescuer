"""Скачивание SeaDronesSee (HF-зеркало dronefreak/SeaDronesSee, YOLO-разметка, CC0).

Качать можно частями, докачка поддерживается (HF-кеш):
    python training/fetch_seadronessee.py --what val              # valid целиком (~1.2 ГБ)
    python training/fetch_seadronessee.py --what shard0           # + первый шард train
    python training/fetch_seadronessee.py --what train            # весь train (~6-7 ГБ)
    python training/fetch_seadronessee.py --what all              # всё выше

Кладёт в data/seadronessee/ (в git не входит, см. .gitignore -> data/).
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = {
    "val": ["data/images/val/*", "data/labels/val/*", "data/data.yaml"],
    "shard0": ["data/images/train/shard_000/*", "data/labels/train/*"],
    "train": ["data/images/train/*", "data/labels/train/*", "data/data.yaml"],
}


def main():
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="val", choices=["val", "shard0", "train", "all"])
    a = ap.parse_args()
    pats = PATTERNS["val"] if a.what == "all" else PATTERNS[a.what]
    if a.what == "all":
        pats = PATTERNS["train"] + PATTERNS["val"]
    out = ROOT / "data" / "seadronessee"
    snapshot_download(
        "dronefreak/SeaDronesSee", repo_type="dataset",
        local_dir=str(out), allow_patterns=pats,
        max_workers=4,
    )
    n = len(list((out / "data" / "images").rglob("*.jpg")))
    print(f"OK: {n} jpg в {out}")


if __name__ == "__main__":
    main()
