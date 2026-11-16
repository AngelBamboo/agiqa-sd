"""Model registry for the benchmark matrix.

Two kinds of entries live here:

* ``pyiqa``  -- published no-reference metrics, run out of the box;
* ``trained`` -- models this repository trains on each database (ResNet-18,
  CLIP+MLP, CoOp, ridge). Their scores come from ``results/runs`` rather than
  from this registry, but the name and family are listed so that the leaderboard
  can merge the two sources without special cases.

``lower_better`` is what the scorer needs to orient a metric; the value is taken
from pyiqa when available and overridden here where pyiqa's metadata disagrees
with the metric's documentation.
"""

# name -> (family, lower_better, note)
PYIQA_MODELS = {
    # classical statistics, no learned parameters
    "niqe": ("classical", True, "natural-scene statistics"),
    "brisque": ("classical", True, "BLIINDS-II style NSS features"),
    "piqe": ("classical", True, "perception-based, no training"),
    # IL-NIQE is excluded from the default matrix: it runs at ~1 image/s and its
    # SVD fails on a handful of generated images, which poisons the CUDA context.
    # "ilniqe": ("classical", True, "integrated local NIQE"),
    # CNN regressors trained on natural-scene IQA
    "cnniqa": ("cnn", False, "small CNN, LIVE-trained"),
    "dbcnn": ("cnn", False, "dual-branch CNN"),
    "nima": ("cnn", False, "NIMA, AVA-trained"),
    "hyperiqa": ("cnn", False, "hypernetwork, content-adaptive"),
    "tres": ("transformer", False, "transformer + relative ranking"),
    # transformer / CLIP-based
    "musiq": ("transformer", False, "multi-scale ViT"),
    "maniqa": ("transformer", False, "multi-dimension attention"),
    "topiq_nr": ("transformer", False, "TOPIQ no-reference"),
    "clipiqa": ("clip", False, "CLIP-IQA"),
    "clipiqa+": ("clip", False, "CLIP-IQA+"),
    "qualiclip": ("clip", False, "quality-aware CLIP"),
    "arniqa": ("transformer", False, "ARNIQA"),
    "liqe": ("clip", False, "vision-language, multi-task"),
    # optional, large: uncomment to include the 7B assessor
    # "qalign": ("llm", False, "Q-Align, mPLUG-Owl2-7B"),
    # "qrealign": ("llm", False, "Q-Realign"),
}

TRAINED_MODELS = {
    "resnet18": ("cnn", "end-to-end ResNet-18"),
    "clip_mlp": ("clip", "frozen CLIP + MLP head"),
    "coop": ("clip", "CoOp-style prompt learning"),
    "ridge": ("linear", "ridge readout on frozen CLIP"),
}


def default_models():
    return list(PYIQA_MODELS)


def lower_better(name, metric=None):
    if name in PYIQA_MODELS:
        return PYIQA_MODELS[name][1]
    if metric is not None:
        return bool(getattr(metric, "lower_better", False))
    return False
