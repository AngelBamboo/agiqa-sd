"""Load every benchmark metric once so that its weights are cached locally.

Run this before the matrix run: the first call to ``pyiqa.create_metric``
downloads weights from HuggingFace, and a slow link otherwise turns a two-hour
benchmark into an overnight one. Each model is instantiated and evaluated on a
single real image; the outcome is printed as OK/FAIL so the log doubles as the
model-availability record for the paper.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from PIL import Image

import pyiqa

MODELS = [
    # classical, no learned weights
    "niqe", "brisque", "piqe", "ilniqe",
    # CNN regressors
    "cnniqa", "dbcnn", "nima", "hyperiqa", "tres",
    # transformer / CLIP-based
    "musiq", "maniqa", "topiq_nr", "clipiqa", "clipiqa+", "qualiclip", "arniqa", "liqe",
]

IMG_DIR = os.environ.get("AGIQA_3K_IMAGES", "D:/AGIQA/data/images")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = sorted(os.listdir(IMG_DIR))[0]
    img = Image.open(os.path.join(IMG_DIR, name)).convert("RGB").resize((512, 512))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1)[None].to(device)

    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for model in MODELS:
        if only and model not in only:
            continue
        t0 = time.time()
        try:
            metric = pyiqa.create_metric(model, device=device)
            with torch.no_grad():
                value = float(metric(x))
            print(f"OK   {model:12s} {time.time() - t0:6.1f}s  score={value:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the log is the deliverable
            print(f"FAIL {model:12s} {time.time() - t0:6.1f}s  "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=1)


if __name__ == "__main__":
    main()
