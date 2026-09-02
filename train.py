"""Training entry point — mirrors the lab notebook cells exactly.

    uv run python train.py --phase baseline    # Phase 1: 50 ep, factory defaults
    uv run python train.py --phase advanced    # Phase 2: 100 ep, custom levers

W&B: reads WANDB_API_KEY from .env (never hardcoded) and logs both runs to the
same project so baseline-vs-advanced curves overlay in one dashboard.
"""
from __future__ import annotations

import argparse
import os

import torch
import wandb
from dotenv import load_dotenv
from ultralytics import YOLO

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')

ROOT_OUTPUT_DIR = "./runs/pose"


def wandb_login() -> None:
    load_dotenv()
    key = os.getenv("WANDB_API_KEY")
    if not key or "paste_your" in key:
        raise ValueError("CRITICAL SECURITY ERROR: 'WANDB_API_KEY' not located in your local .env file!")
    wandb.login(key=key)


def enable_webcam_blur_augmentation(p_blur: float = 0.15) -> None:
    """Implement the lab's 'webcam blur simulation' for real.

    `blur=0.15` is NOT a valid Ultralytics train argument (the notebook's line
    crashes with `SyntaxError: 'blur' is not a valid YOLO argument`). The
    correct mechanism is the Albumentations hook inside Ultralytics' train
    pipeline, so we amplify its motion/defocus blur probability instead.
    Pixel-level only — keypoint coordinates are untouched.
    """
    import albumentations as A
    import ultralytics.data.augment as aug

    orig_init = aug.Albumentations.__init__

    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        self.contains_spatial = False
        self.transform = A.Compose([
            A.Blur(p=p_blur, blur_limit=(3, 7)),
            A.MedianBlur(p=p_blur / 3, blur_limit=(3, 7)),
        ])
        print(f"[webcam-sim] Albumentations blur augmentation active (p={p_blur})")

    aug.Albumentations.__init__ = patched_init


def find_last_checkpoint(run_name: str):
    """Newest last.pt for this run, wherever ultralytics nested it."""
    import glob
    hits = [p for p in glob.glob(f"runs/**/{run_name}/weights/last.pt", recursive=True)]
    return max(hits, key=os.path.getmtime) if hits else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "advanced"], default="baseline")
    parser.add_argument("--device", default=0, help="GPU index (default 0)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the last per-epoch checkpoint")
    args = parser.parse_args()

    print(f"CRITICAL RESOURCE ALLOCATION -> Active Hardware Device: {device.upper()}")
    wandb_login()

    if args.resume:
        run_name = "train" if args.phase == "baseline" else "train_advanced"
        last = find_last_checkpoint(run_name)
        if last:
            print(f"RESUMING {args.phase} from checkpoint: {last}")
            wandb.init(project="hand-pose-estimation", name=f"{args.phase}-resumed")
            YOLO(last).train(resume=True, device=args.device)
            return
        print(f"No checkpoint found for '{run_name}' — starting fresh.")
    # Open the W&B run ourselves so the dashboard gets a clean project/run name
    # (Ultralytics' callback reuses an existing wb.run instead of creating one
    # named after the output directory path).
    wandb.init(project="hand-pose-estimation", name=args.phase)

    if args.phase == "baseline":
        print("Running Phase 1: 50-Epoch Baseline Pipeline (Factory Defaults)")
        model = YOLO("yolo26n-pose.pt")
        model.train(
            data="hand-keypoints.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            device=args.device,
            deterministic=True,
            save=True,
            project=ROOT_OUTPUT_DIR,
            name="train",
        )
    else:
        print("Running Phase 2: Advanced Custom Optimization Sequence")
        enable_webcam_blur_augmentation(p_blur=0.15)
        model = YOLO("yolo26n-pose.pt")
        model.train(
            data="hand-keypoints.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            device=args.device,
            deterministic=True,
            save=True,
            patience=20,

            # --- Advanced Optimization Levers ---
            optimizer="AdamW",          # adaptive gradient scaling
            lr0=0.002,                  # initial step size tuned for AdamW
            cos_lr=True,                # smooth cosine-annealing decay profile
            warmup_epochs=4.0,          # extended gradient stabilization phase

            # --- Loss Coordinate Regularization ---
            pose=18.0,                  # amplify penalty for misplaced finger joints
            box=5.0,                    # de-emphasize bounding box bounds slightly

            # --- Webcam Simulation Augmentations ---
            # blur is applied via enable_webcam_blur_augmentation() above —
            # `blur=` is not a valid Ultralytics argument (course notebook bug).
            scale=0.6,                  # regularize against variable camera distances

            project=ROOT_OUTPUT_DIR,
            name="train_advanced",
        )


if __name__ == "__main__":
    main()
