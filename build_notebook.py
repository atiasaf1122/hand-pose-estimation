"""Builds hand_pose_lab.ipynb — the lab notebook, executed with real results.

    python build_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except (AttributeError, OSError):
    pass

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "hand_pose_lab.ipynb"

cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda s: cells.append(new_code_cell(s))

md("""# Hand Pose Estimation with YOLOv26-Pose
### Fine-tuning a generic body-pose detector into a 21-joint hand tracker

**Task.** The stock `yolo26n-pose` model is trained on COCO-pose: it finds a **person**
and marks **17 body keypoints**. A whole hand is represented by a *single* wrist point —
it has no concept of fingers or knuckles. We fine-tune it on the `hand-keypoints`
dataset so that it detects **hands** (`kpt_shape [17,3] -> [21,3]`) and tracks all
21 joints in real time from a webcam.

**Experiment.** Two training runs on identical data, seed and image size — the *only*
difference is the optimisation recipe:

| | baseline | advanced |
|---|---|---|
| optimizer | auto (SGD) | AdamW, `lr0=0.002` |
| LR schedule | linear decay | cosine annealing, `warmup=4` |
| loss weights | `pose=12, box=7.5` | `pose=18, box=5` |
| augmentation | defaults | `+ blur(p=0.15)`, `scale=0.6` |
| budget | 50 epochs | 100 epochs, `patience=20` |

**Headline result.** The advanced recipe did **not** beat the defaults
(0.867 vs **0.902** pose mAP50). Section 6 shows *why*, with a controlled experiment.""")

md("""## 1. Environment

Dependencies are locked with **uv** (`pyproject.toml` + `uv.lock`); run `uv sync` before
this notebook. Two notes on the environment that mattered in practice:

* **Python is pinned to 3.12** — `cv2` has a known crash with 3.14 inside DataLoader workers.
* **`pyproject.toml` adds the PyTorch CUDA index** — PyPI ships a CPU-only build on
  Windows, which would have made training ~20x slower.""")

code("""import os, sys, time
import cv2, torch
from ultralytics import YOLO

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
print(f"CRITICAL RESOURCE ALLOCATION -> Active Hardware Device: {device.upper()}")
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")""")

md("""## 2. Credentials — Weights & Biases

The API key is loaded from a local `.env` (git-ignored), never hardcoded. A public repo
with a committed key would let anyone write to the account.

```env
WANDB_API_KEY=your_private_key_here
```

One non-obvious step: the Ultralytics W&B integration is **disabled by default**, and its
callback derives the project name from the output *path* (giving `.-runs-pose`). We
therefore enable it and open the run ourselves, so both trainings land in one comparable
dashboard.""")

code("""from dotenv import load_dotenv
import wandb
from ultralytics.utils import SETTINGS

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    raise ValueError("CRITICAL SECURITY ERROR: 'WANDB_API_KEY' not located in your local .env file!")

SETTINGS.update({"wandb": True})          # integration is OFF by default
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    wandb.login(key=WANDB_API_KEY)      # quiet: its banner echoes local file paths
print("W&B login OK")
# wandb.init(project="hand-pose-estimation", name="baseline")   # done inside train.py""")

md("""## 3. Baseline record — how the *generic* model handles a hand

Before touching any weights we record the starting point. `detect.py` loads the stock
`yolo26n-pose.pt` when no fine-tuned checkpoint is present, runs the webcam for 8 seconds
and writes `runs/pose/predict/failed_attempt.mp4`.

```bash
uv run python detect.py --weights none.pt --output runs/pose/predict/failed_attempt.mp4
```

**What it actually does** — and this is the precise framing: the generic model does *not*
malfunction. It detects `person` at ~0.89 confidence and draws the COCO body skeleton
(eyes, nose, ears, shoulders). Each hand receives exactly **one** keypoint — the wrist —
because that is the maximum hand resolution COCO-pose has. The "failure" is a
**task mismatch**, not a broken model.""")

code("""# The recorder is a script (it needs a real camera + window loop), not a notebook cell.
# Reproduce with:
#   uv run python detect.py --weights none.pt --output runs/pose/predict/failed_attempt.mp4
generic = YOLO("yolo26n-pose.pt")
print("generic model kpt_shape :", generic.model.yaml.get("kpt_shape"), "<- 17 COCO body points")
print("generic model classes   :", generic.names)""")

md("""## 4. Training — two recipes, one variable

Both runs are launched from `train.py`, which mirrors the lab configuration exactly.
They ran **in parallel on two RTX 3090s**.

```bash
uv run python train.py --phase baseline --device 0    # 50 epochs, factory defaults
uv run python train.py --phase advanced --device 1    # 100 epochs, custom levers
```

> **A bug in the provided configuration.** The lab specifies `blur=0.15` as a training
> argument, but no such argument exists in Ultralytics — the call raises
> `SyntaxError: 'blur' is not a valid YOLO argument`. We implemented the stated intent
> ("webcam blur simulation") the supported way: injecting `A.Blur`/`A.MedianBlur` into the
> Albumentations pipeline. These are pixel-level transforms, so keypoint coordinates are
> untouched — safe for pose labels.""")

code('''# --- the exact configuration used (see train.py for the runnable version) ---
BASELINE = dict(data="hand-keypoints.yaml", epochs=50, imgsz=640, batch=16,
                deterministic=True, save=True, project="./runs/pose", name="train")

ADVANCED = dict(data="hand-keypoints.yaml", epochs=100, imgsz=640, batch=16,
                deterministic=True, save=True, patience=20,
                optimizer="AdamW", lr0=0.002, cos_lr=True, warmup_epochs=4.0,
                pose=18.0, box=5.0, scale=0.6,
                project="./runs/pose", name="train_advanced")

for k in sorted(set(BASELINE) | set(ADVANCED)):
    b, a = BASELINE.get(k, "-"), ADVANCED.get(k, "-")
    if b != a:
        print(f"{k:16s} baseline={str(b):22s} advanced={a}")''')

md("""## 5. Results""")

code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(11,4),'axes.grid':True,'grid.alpha':.3,
                     'axes.spines.top':False,'axes.spines.right':False,'font.size':10})

runs = {'baseline': pd.read_csv('results/baseline_results.csv'),
        'advanced': pd.read_csv('results/advanced_results.csv')}
COL = {'baseline':'#2a9d8f', 'advanced':'#e76f51'}

rows = []
for tag, df in runs.items():
    df.columns = [c.strip() for c in df.columns]
    best = df.loc[df['metrics/mAP50-95(P)'].idxmax()]
    rows.append({
        'run': tag,
        'epochs': int(df['epoch'].max()),
        'best epoch': int(best['epoch']),
        'box_loss': round(best['train/box_loss'], 4),
        'pose_loss': round(best['train/pose_loss'], 4),
        'cls_loss': round(best['train/cls_loss'], 4),
        'Pose mAP50': round(best['metrics/mAP50(P)'], 4),
        'Pose mAP50-95': round(best['metrics/mAP50-95(P)'], 4),
        'Box mAP50': round(best['metrics/mAP50(B)'], 4),
    })
summary = pd.DataFrame(rows).set_index('run')
print(summary.to_string())""")

code("""fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for tag, df in runs.items():
    ax[0].plot(df['epoch'], df['metrics/mAP50(P)'],    color=COL[tag], label=tag)
    ax[1].plot(df['epoch'], df['metrics/mAP50-95(P)'], color=COL[tag], label=tag)
    ax[2].plot(df['epoch'], df['val/pose_loss'],       color=COL[tag], label=tag)
ax[0].set_title('Pose mAP@50 (validation)');    ax[0].set_xlabel('epoch')
ax[1].set_title('Pose mAP@50-95 (validation)'); ax[1].set_xlabel('epoch')
ax[2].set_title('Validation pose loss');        ax[2].set_xlabel('epoch')
for a in ax: a.legend()
plt.tight_layout(); plt.show()

for tag, df in runs.items():
    i = df['val/pose_loss'].idxmin()
    print(f"{tag:9s} min val/pose_loss = {df['val/pose_loss'].min():.4f} at epoch {int(df.loc[i,'epoch'])}"
          f"  (last epoch = {int(df['epoch'].max())})")""")

md("""**Reading the curves.**

* The validation loss minimum sits at the **final epoch of both runs** — neither model
  overfit, and neither tripped the `patience=20` early stop. Both were still improving
  when their epoch budget ran out; the runs are *compute-limited, not convergence-limited*.
* The advanced run's **training** `pose_loss` looks higher, but that is an artefact of the
  loss weights: it is multiplied by 18 versus 12 (a 1.5x ratio, exactly what the gap shows).
  Weighted losses are not comparable across runs — the validation mAP is.
* The step-change near the end of each run is `close_mosaic`: Ultralytics disables mosaic
  augmentation for the last 10 epochs, so the training distribution suddenly gets easier.""")

md("""## 6. Why did the "advanced" recipe lose?

The advanced run trains with blur to simulate webcam conditions — but the validation set is
**clean**. We are measuring a model trained for noisy input on a noiseless benchmark: it is
running the race wearing the training weights.

To test that explanation we built the missing evaluation: a copy of all 7,992 validation
images degraded with the *same* blur kernel range used in training (`eval_blur.py`).""")

code("""blur = pd.read_csv('results/blur_stress_test.csv')
piv = blur.pivot(index='model', columns='val_set', values='pose_mAP50')[['clean','blurred']]
piv['drop'] = (piv['clean'] - piv['blurred']).round(4)
piv['drop %'] = (100 * piv['drop'] / piv['clean']).round(1)
print(piv.to_string())

fig, ax = plt.subplots(figsize=(7,4))
x = np.arange(2); w = 0.35
for i,(m,c) in enumerate([('baseline',COL['baseline']),('advanced',COL['advanced'])]):
    ax.bar(x + (i-0.5)*w, [piv.loc[m,'clean'], piv.loc[m,'blurred']], w, label=m, color=c)
ax.set_xticks(x); ax.set_xticklabels(['clean validation','blurred validation'])
ax.set_ylabel('Pose mAP@50'); ax.set_ylim(0.75, 0.95); ax.legend()
ax.set_title('Robustness to input degradation')
plt.tight_layout(); plt.show()""")

md("""**Result: the augmentation worked — the recipe still lost.**

The advanced model degrades **less** under blur (-3.1 points vs -4.2, i.e. 3.5% vs 4.7%
relative), so the blur augmentation did buy measurable robustness. But the baseline starts
so far ahead that it remains better on *both* sets.

The honest conclusion: the blur term achieved its stated goal, while the rest of the recipe
(AdamW at `lr0=0.002`, the loss re-weighting) cost more accuracy than the robustness was
worth **on this task**. A tuned default beat an aggressive hand-tuned configuration.""")

md("""## 7. Verification on live webcam

Training happened on the desktop (2x RTX 3090, no camera); the recordings were made on a
laptop (RTX 4050) — the hybrid workflow the lab describes. `uv.lock` reproduces the exact
environment there, and only the `best.pt` files travel.

```bash
uv run python detect.py --weights weights/baseline_best.pt --output runs/pose/predict/successful_attempt.mp4
```

All three models were recorded back-to-back on the same scene, giving a third independent
comparison under real conditions:""")

code("""live = pd.read_csv('results/webcam_live_test.csv')
print(live.to_string(index=False))""")

md("""The ordering holds a third time: **baseline 0.94 > advanced 0.92** mean confidence on
live webcam footage. Together with clean validation (0.902 vs 0.867) and blurred validation
(0.860 vs 0.837), the conclusion is supported from three independent angles.

**The two submitted clips**

| file | weights | what it shows |
|---|---|---|
| `runs/pose/predict/failed_attempt.mp4` | generic `yolo26n-pose.pt` | detects `person`, body skeleton on the face, **one wrist point per hand** |
| `runs/pose/predict/successful_attempt.mp4` | fine-tuned `best.pt` | a `hand` box per hand at 0.94 confidence, all **21 joints** tracked, holds through a closed fist |""")

md("""## 8. Conclusions

1. **Fine-tuning succeeded.** From one wrist point per hand to 21 tracked joints at
   0.902 pose mAP50 / 0.762 mAP50-95.
2. **The advanced recipe did not.** It lost on clean validation, on blurred validation and
   on live webcam. Reporting that honestly is more useful than reporting a win.
3. **The blur augmentation specifically did work** — it cut the degradation under blur from
   4.7% to 3.5% relative. The component was sound; the surrounding recipe was too aggressive.
4. **Neither run overfit.** Validation loss was still falling at the final epoch of both,
   so the limiting factor was the epoch budget, not generalisation.
5. **The provided configuration contained an invalid argument** (`blur=0.15`), which was
   diagnosed and re-implemented through the supported Albumentations hook so the intended
   augmentation was still applied.

**If continued:** re-run the advanced recipe with `lr0` lowered toward the SGD-equivalent
regime and the loss weights left at their defaults, to isolate which lever caused the loss —
the current experiment changes five at once.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})

print(f"assembled {len(cells)} cells; executing ...")
try:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    print("executed OK")
except Exception as e:
    print("WARN execute:", e)


# Keep the notebook machine-independent: drop console lines that embed an
# absolute filesystem path (tool banners print the local home directory).
import re as _re
_ABS_PATH = _re.compile(r'[A-Za-z]:[\/]|/home/')
def _portable(lines):
    return [ln for ln in lines if not _ABS_PATH.search(ln)]
for _c in nb.cells:
    for _o in _c.get('outputs', []):
        if 'text' in _o:
            _t = _o['text']
            _o['text'] = _portable(_t) if isinstance(_t, list) else chr(10).join(_portable(_t.split(chr(10))))

nbf.write(nb, str(OUT))
print("wrote", OUT)
