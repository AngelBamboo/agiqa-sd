"""Path configuration.

Every script reads its locations from here so that the repository can be run
from any directory. Override with environment variables if your data lives
elsewhere:

    AGIQA_3K       AGIQA-3K root      (contains data.csv and images/)
    AGIQA_2023     AIGCIQA2023 root   (contains data.csv and allimg/)
    AGIQA_1K       AGIQA-1K root      (contains AIGC_all.csv and file/)
    AGIQA_FEATURES directory for cached CLIP features
    AGIQA_RUNS     directory for experiment records
    AGIQA_FIGURES  output directory for the paper figures
"""
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(env, default):
    value = os.environ.get(env)
    if value:
        return value
    return os.path.join(_HERE, default)


DATA_3K = _path("AGIQA_3K", os.path.join("data", "agiqa3k"))
DATA_2023 = _path("AGIQA_2023", os.path.join("data", "aigciqa2023"))
DATA_1K = _path("AGIQA_1K", os.path.join("data", "agiqa1k"))
FEATURES = _path("AGIQA_FEATURES", "features")
RUNS = _path("AGIQA_RUNS", "runs")
FIGURES = _path("AGIQA_FIGURES", "figs")

CLIP_3K = os.path.join(FEATURES, "agiqa3k_ViT-B-32_openai.npy")
CLIP_2023 = os.path.join(FEATURES, "aigciqa2023_ViT-B-32_openai.npy")

# output directories are created on import so no script has to guess
for _d in (FEATURES, RUNS, FIGURES):
    os.makedirs(_d, exist_ok=True)
