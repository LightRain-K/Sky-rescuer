from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dfire.yaml"


def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(ROOT / "runs"),
        name="dfire_yolo11n",
        patience=10,
    )
    metrics = model.val(data=str(DATA), split="test")
    print(f"mAP50-95 (test): {metrics.box.map:.4f}")
    print(f"mAP50 (test): {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
