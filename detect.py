"""Live webcam hand-pose inference + automatic 8-second recording.

Phase 1 (baseline, generic COCO body-pose weights -> the "failed" record):
    uv run python detect.py --output runs/pose/predict/failed_attempt.mp4

Phase 2 (after training, fine-tuned hand weights -> the "successful" record):
    uv run python detect.py --output runs/pose/predict/successful_attempt.mp4

If ./runs/pose/train/weights/best.pt exists it is used automatically;
otherwise the generic pretrained `yolo26n-pose.pt` is loaded.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import torch
from ultralytics import YOLO

RECORD_SECONDS = 8.0
CUSTOM_WEIGHTS = "./runs/pose/train/weights/best.pt"

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')


def gui_available() -> bool:
    """Probe once whether this OpenCV build can actually open windows.

    The headless wheel ships the same `cv2` module but raises on any highgui
    call, so the only reliable test is to try creating a window.
    """
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def record_inference_run(weights_path: str, output_filename: str,
                         camera_index: int = 0) -> None:
    """Run live camera tracking and record an 8-second annotated sample video."""
    if os.path.exists(weights_path):
        print(f"Loading custom optimized model weights from: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"Custom weights absent at '{weights_path}'. Loading generic base pretrained model...")
        model = YOLO("yolo26n-pose.pt")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Cannot interface with webcam at index {camera_index}.")
        print("Try a different index, e.g.  --camera 1")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    show_preview = gui_available()
    if not show_preview:
        print("\n" + "=" * 62)
        print("NOTE: this OpenCV build has no GUI support - no live preview.")
        print("Recording still works. A per-second detection report is printed")
        print("below so you can judge the take without watching it live.")
        print("=" * 62 + "\n")

    # Warm-up: burn a few frames so CUDA graph/kernel init happens OUTSIDE the
    # timed window. (Timing it here would under-estimate the rate badly — the
    # first frames are ~5x slower — which is why the real fps is measured over
    # the recording itself, below.)
    print("Warming up the model...")
    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            break
        model.predict(frame, imgsz=640, device=device, verbose=False, conf=0.25)

    print(f"VIDEO RECORDER STARTED -> Target Path: {output_filename}")
    print(f"Recording closes automatically after {RECORD_SECONDS:.0f} seconds. Press 'q' to abort early...")

    # Frames are buffered and written at the end with the measured capture rate,
    # so the saved clip plays back in real time regardless of hardware speed.
    buffered = []
    total_frames = 0
    detected_frames = 0
    conf_sum = 0.0
    next_tick = 1.0

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= RECORD_SECONDS:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV numpy frames are BGR - exactly what Ultralytics expects for
        # numpy input, so the frame is passed through without conversion and
        # the annotated output stays in BGR for the writer and the preview.
        results = model.predict(frame, imgsz=640, device=device, verbose=False, conf=0.25)
        r = results[0]

        total_frames += 1
        n_det = 0 if r.boxes is None else len(r.boxes)
        if n_det > 0:
            detected_frames += 1
            conf_sum += float(r.boxes.conf.max())

        annotated_frame = r.plot()
        buffered.append(annotated_frame)

        # Console heartbeat: one line per second of recording, so a headless
        # run still tells you whether the hand is being picked up.
        if elapsed >= next_tick:
            status = f"{n_det} detection(s)" if n_det else "NOTHING DETECTED"
            print(f"  [{int(elapsed)}s] {status}")
            next_tick += 1.0

        if show_preview:
            cv2.imshow("YOLOv26 Live Stream Video Recorder", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture_seconds = time.time() - start_time
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    real_fps = total_frames / max(capture_seconds, 1e-6)
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'),
                          real_fps, (frame_width, frame_height))
    for f in buffered:
        out.write(f)
    out.release()

    hit_rate = (detected_frames / total_frames * 100) if total_frames else 0.0
    mean_conf = (conf_sum / detected_frames) if detected_frames else 0.0

    print(f"\nRECORDING COMPLETE: File saved successfully to {output_filename}")
    print("-" * 62)
    print(f"  frames written        : {total_frames}")
    print(f"  true playback rate    : {real_fps:.1f} FPS  ({capture_seconds:.1f}s captured)")
    print(f"  frames with detection : {detected_frames}  ({hit_rate:.0f}%)")
    print(f"  mean confidence       : {mean_conf:.2f}")
    print("-" * 62)

    if hit_rate >= 80:
        print("  VERDICT: strong, consistent tracking - good 'successful' take.")
    elif hit_rate >= 30:
        print("  VERDICT: intermittent tracking. Fine for a 'failure' clip;")
        print("           re-shoot with better light if this was meant to succeed.")
    else:
        print("  VERDICT: almost nothing tracked - textbook 'failed_attempt' clip.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv26-Pose webcam recorder")
    parser.add_argument("--weights", default=CUSTOM_WEIGHTS,
                        help="path to fine-tuned weights (falls back to yolo26n-pose.pt)")
    parser.add_argument("--output", default="./runs/pose/predict/failed_attempt.mp4",
                        help="output .mp4 path")
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam index (try 1 if the default camera fails)")
    args = parser.parse_args()

    print(f"CRITICAL RESOURCE ALLOCATION -> Active Hardware Device: {device.upper()}")
    record_inference_run(weights_path=args.weights,
                         output_filename=args.output,
                         camera_index=args.camera)