"""2x2 stress test: both trained models x {clean, blurred} validation sets."""
import os

os.environ["WANDB_MODE"] = "disabled"      # side experiment — keep the dashboard clean
from ultralytics import YOLO

W = {
    "baseline": "runs/pose/runs/pose/train/weights/best.pt",
    "advanced": "runs/pose/runs/pose/train_advanced/weights/best.pt",
}
print(f"{'model':10s} {'set':8s} {'pose mAP50':>11s} {'pose mAP50-95':>14s}")
for name, w in W.items():
    for tag, data in [("clean", "hand-keypoints.yaml"), ("blurred", "hand-keypoints-blurval.yaml")]:
        r = YOLO(w).val(data=data, device=1, batch=32, workers=0, half=True,
                        verbose=False, plots=False)
        print(f"{name:10s} {tag:8s} {r.pose.map50:11.4f} {r.pose.map:14.4f}", flush=True)
