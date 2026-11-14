# Source Confounding in AI-Generated Image Quality Assessment

Code, splits, and per-image predictions for the paper:

> **Source Confounding in AI-Generated Image Quality Assessment: Diagnosis, Protocol, and a Capacity-Based Remedy**

The claim of the paper is narrow and testable: on two public AI-generated image
quality databases, models trained with the standard random split almost entirely
learn *which generator produced an image*, not *how good the image is*. Under
leave-one-generator-out (LOGO) evaluation the same models lose roughly a third of
their correlation, and — once the reliability of the subjective scores is accounted
for — they exploit only about half of the ranking signal the annotations allow.

This repository contains everything needed to reproduce every number, table and
figure in the paper, plus the full per-image predictions so that the analysis can
be re-run without retraining.

---

## Main results

Spearman rank correlation (SRCC) on AGIQA-3K. LOGO is the equal-weight mean over
the six held-out generators (SD 1.5, Midjourney, GLIDE, AttnGAN, SDXL, DALL·E 2).

| Protocol | Model | SRCC | Backbone |
|---|---|---:|---|
| Random 80/20 | ResNet-18 | 0.830 ± 0.013 | trained from ImageNet init |
| Random 80/20 | CLIP ViT-B/32 + MLP | 0.832 ± 0.028 | frozen features |
| Random 80/20 | CoOp (16 ctx tokens) | 0.846 ± 0.020 | frozen features + prompts |
| Random 80/20 | Ridge (α = 1000) | **0.849 ± 0.019** | frozen features |
| Prompt-grouped | ResNet-18 | 0.813 ± 0.013 | trained |
| LOGO | ResNet-18 | 0.523 | trained |
| LOGO | CLIP ViT-B/32 + MLP | 0.544 | frozen features |
| LOGO | CoOp | 0.556 | frozen features + prompts |
| LOGO | Ridge (α = 1000) | **0.579** | frozen features |

Diagnostics on the random split, where all three model families agree
(CLIP features, seed 0):

| Quantity | Value | Meaning |
|---|---:|---|
| overall SRCC | 0.848 | ranking over the whole test set |
| within-source SRCC | 0.518 | equal-weight mean over the six generators |
| between-source SRCC | 1.000 | ranking of the six generator means |
| variance explained by source identity | 57.5 % | a property of the database, model-independent |
| source-only control / CoOp SRCC | 50 % | nearest-centroid classifier that never sees quality |

Additional evidence in the paper: a second database (AIGCIQA2023: random 0.815,
LOGO 0.421), cross-database transfer (AGIQA-3K → AGIQA-1K 0.569, → AIGCIQA2023
0.600), a capacity study (2 epochs already reach 0.824), a split-half reliability
analysis with Spearman–Brown correction, and six debiasing attempts of which most
fail — reported as they are.

---

## Repository layout

```
agiqa/                  shared package
  config.py             all paths, overridable by environment variable
  data.py               dataset readers, the three split protocols
  models.py             ResNet-18 quality net, MLP head, gradient reversal layer
  metrics.py            SRCC / PLCC / KRCC helpers

train.py                ResNet-18 training (random / prompt / logo, augmentation,
                        adversarial head, ranking loss, per-image dump)
head_eval.py            linear / MLP readout on frozen CLIP features (WSR, reliability weights)
coop_agiqa.py           CoOp-style prompt learning on frozen CLIP
nested_ridge.py         ridge readout with nested alpha selection (3K and 2023)
alpha_sweep.py          sensitivity of the ridge readout to its penalty
reliability.py          split-half reliability and Cronbach alpha of the 2023 MOS
capacity_study.py       how fast the shortcut is learned (epoch budget sweep)
source_only.py          nearest-generator-centroid control (no quality signal)
score_baselines.py      NIQE / BRISQUE / MUSIQ / CLIP-IQA / TOPIQ reference scores
extract_features.py     cache CLIP ViT-B/32 features for AGIQA-3K
extract_2023.py         cache CLIP ViT-B/32 features for AIGCIQA2023
eval_2023.py            protocol comparison on AIGCIQA2023
cross_eval.py           cross-database transfer
make_figs.py            regenerate every figure from results/runs/

results/summary.csv     the main table above, machine readable
results/runs/           one JSON record per experiment (60 files)
results/predictions/    per-image predictions (12 CSV files)
data/                   put the datasets here (not shipped, see below)
figs/                   created by make_figs.py
```

---

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended but not required — every
frozen-feature experiment also runs on CPU, just slower.

```bash
pip install -r requirements.txt
```

The CLIP experiments use [open_clip](https://github.com/mlfoundations/open_clip)
with the `ViT-B-32` / `openai` weights. Feature extraction is the only step that
downloads anything.

---

## Data

The images are not redistributed here. Download them from the original sources and
unpack them into `data/` (or point the environment variables anywhere else):

| Directory | Dataset | Source |
|---|---|---|
| `data/agiqa3k/` | AGIQA-3K | https://github.com/lcysyzxdxc/AGIQA-3K |
| `data/aigciqa2023/` | AIGCIQA2023 | https://github.com/wanghaining/2023-AIGC — the images are linked from that repository's README |
| `data/agiqa1k/` | AGIQA-1K | https://github.com/lcysyzxdxc/AGIQA-1K |

Expected layout:

```
data/agiqa3k/         data.csv  images/
data/aigciqa2023/     data.csv  allimg/          (per-image rater scores in the JSON index)
data/agiqa1k/         AIGC_all.csv  file/
```

Override the locations if needed:

```bash
export AGIQA_3K=/path/to/agiqa3k
export AGIQA_2023=/path/to/aigciqa2023
export AGIQA_1K=/path/to/agiqa1k
export AGIQA_FEATURES=/path/to/features
export AGIQA_RUNS=/path/to/runs
```

On Windows, `set VAR=value` in `cmd` or `$env:VAR="value"` in PowerShell.

---

## Reproduction

The commands below assume AGIQA-3K is in place. `--seed` values used in the paper
are 0, 1, 2.

**1. Cache CLIP features** (once, ~10 min on a 4070-class GPU)

```bash
python extract_features.py
```

**2. Standard random split**

```bash
python train.py     --protocol random --seed 0
python head_eval.py --protocol random --seed 0 --dump results/predictions/preds_clip_random.csv
python coop_agiqa.py --protocols random --seeds 0,1,2
```

**3. Prompt-grouped split**

```bash
python train.py --protocol prompt --seed 0
```

**4. Leave-one-generator-out**, once per held-out generator

```bash
for g in sd1.5 midjourney AttnGAN glide xl2.2 DALLE2; do
  python train.py     --protocol logo --holdout "$g" --seed 0
  python head_eval.py --protocol logo --holdout "$g" --seed 0
done
python nested_ridge.py                    # nested alpha selection, all six folds
python alpha_sweep.py                     # penalty sensitivity analysis
for g in sd1.5 midjourney AttnGAN glide xl2.2 DALLE2; do
  python source_only.py --holdout "$g" --out auto   # source-inference control
done
```

Generator labels follow the AGIQA-3K release: `sd1.5` = SD 1.5, `xl2.2` = SDXL 2.2,
`DALLE2` = DALL·E 2, `glide` = GLIDE, `midjourney` = Midjourney. They are listed in
`GENERATORS` in `agiqa/data.py`.

**5. Second database and cross-database transfer**

```bash
python extract_2023.py
python eval_2023.py --seeds 0,1,2     # random + six LOGO folds, three seeds
python nested_ridge.py --dataset 2023
python reliability.py                 # split-half and Cronbach alpha of the MOS
python cross_eval.py --weights <checkpoint>.pt
```

**6. Figures**

```bash
python make_figs.py          # writes to $AGIQA_FIGURES, default ./figs
```

Every script writes a JSON record into `$AGIQA_RUNS`; the files already committed
under `results/runs/` are those records. `make_figs.py` reads them, plus the
dataset CSVs for the mean-opinion-score panel, so all six figures can be rebuilt
from the archive without retraining or re-extracting features.

---

## Notes on the protocols

* **random** — 80/20 over images, the split used by essentially all published
  AGIQA work. Images from the same generator appear in both halves.
* **prompt** — grouped by prompt text, so no prompt is shared between train and
  test. Generator identity still overlaps.
* **logo** — one generator held out entirely. Prompt overlap is *not* removed:
  prompts are shared across generators, so the same prompt can still appear in
  both halves, paired with a different generator's image. Excluding both at once
  would need a joint prompt×generator holdout, which we did not run and do not
  claim.

The `within` / `between` / `overall` figures are diagnostic summaries, not a
variance decomposition: SRCC is a rank statistic, so the overall value is not a
weighted sum of the other two.

Hyperparameters (ridge α, CoOp learning rate) are selected by nested validation
inside the training sources only — the held-out generator never participates in
model selection. `nested_ridge.py` prints the α chosen per fold.

---

## Citation

```bibtex
@article{agiqa_source_confounding,
  title   = {Source Confounding in AI-Generated Image Quality Assessment:
             Diagnosis, Protocol, and a Capacity-Based Remedy},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`. The datasets keep their own licenses; please cite the
original AGIQA-3K, AIGCIQA2023 and AGIQA-1K papers if you use them.
