# The Pooled SRCC Measures Generators

Code, splits, and per-image predictions for the paper:

> **The Pooled SRCC Measures Generators: Diagnosing Source Confounding in AI-Generated Image Quality Assessment**

The claim of the paper is narrow and testable: on two public AI-generated image
quality databases, the pooled random-split SRCC that almost all published numbers
rest on is dominated by the ordering of generators. The same models order the six
generators of AGIQA-3K perfectly (between-generator SRCC 1.000) while ranking
images *within* a generator at only 0.52–0.58. A paired experiment on identical
test images gives mean seen-minus-unseen differences of +0.019 (ResNet-18),
−0.016 (CLIP+MLP) and +0.014 (ridge) — an order of magnitude below the 0.30 gap —
although individual generators range from −0.25 to +0.12 (Table 5 of the paper).
The familiar 0.83 → 0.54 drop is therefore a property of the statistic rather than
a loss of ability. Against an estimated reference ceiling of 0.90–0.99 under the
stated reliability model, models reach 45–77%.

The repository contains the code, the split definitions, one JSON record per run
and the per-image predictions behind every number, table and figure in the paper.
The committed predictions cover seed 0 for most protocols; the other seeds are
archived as summary records in `results/runs/`, so they can be inspected and
re-checked but not re-analysed per image without retraining.

---

## Main results

Spearman rank correlation (SRCC) on AGIQA-3K. LOGO is the equal-weight mean over
the six held-out generators (SD 1.5, Midjourney, GLIDE, AttnGAN, SDXL, DALL·E 2).

| Protocol | Model | SRCC | Backbone |
|---|---|---:|---|
| Random 80/20 | ResNet-18 | 0.830 ± 0.013 | trained from ImageNet init |
| Random 80/20 | CLIP ViT-B/32 + MLP | 0.832 ± 0.028 | frozen features |
| Random 80/20 | CoOp (16 ctx tokens) | 0.846 ± 0.025 | frozen features + prompts |
| Random 80/20 | Ridge (α = 1000) | **0.849 ± 0.023** | frozen features |
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
| source-inference baseline / CoOp SRCC | 50 % | nearest-centroid label, no image-level quality regressor |
| cross-source share of Kendall's net agreement | 83.4 % | pairs joining two generators are 75.8 % of the test set |

Additional evidence in the paper: a second database (AIGCIQA2023: random 0.815,
LOGO 0.421, paired seen-minus-unseen +0.037), cross-database transfer (AGIQA-3K →
AGIQA-1K 0.569, → AIGCIQA2023 0.600), a training-budget study (a ResNet-18 reaches
0.821 after 2 epochs, seed 0), a reliability estimate based on the released
per-image rating spread and the reported participant count (ICC(1,k) with
k = 11/21/31 sensitivity), and six debiasing attempts of which most fail — reported
as they are.

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
reliability.py          ceiling on correlation from the released per-image rating spread
probe.py                linear probe: how decodable is the generator from the features
capacity_study.py       how fast the shortcut is learned (epoch budget sweep)
source_only.py          nearest-generator-centroid baseline (no image-level quality regressor)
paired_source.py        seen-source vs unseen-source on identical test images (frozen features)
paired_resnet.py        the same control on the end-to-end ResNet-18
report.py               print pooled / within-source / between-source SRCC for any prediction file
score_baselines.py      NIQE / BRISQUE / MUSIQ / CLIP-IQA / TOPIQ reference scores
extract_features.py     cache CLIP ViT-B/32 features for AGIQA-3K
extract_2023.py         cache CLIP ViT-B/32 features for AIGCIQA2023
eval_2023.py            protocol comparison on AIGCIQA2023
cross_eval.py           cross-database transfer
make_figs.py            regenerate every figure from results/runs/

results/summary.csv     the main table above, machine readable
results/runs/           one JSON record per experiment (136 files), including the
                        paired seen/unseen runs for all three test splits, with and
                        without source-balanced targets, and the AIGCIQA2023 paired run
results/predictions/    per-image predictions (51 CSV files: random split, six LOGO
                        folds for ResNet-18 / CLIP+MLP / CoOp / ridge, and the 2023
                        protocol runs). Seed 0 unless the file name or a `seed`
                        column says otherwise.
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
data/aigciqa2023/     data.csv  allimg/          (aggregate MOS only; no per-rater scores)
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
python paired_source.py --model ridge     # seen-source vs unseen-source, paired
python paired_source.py --model mlp --seeds 0,1,2
python paired_source.py --model ridge --normalize source --split_seed 0   # source-balanced targets
python paired_source.py --model mlp   --normalize source --split_seed 0
python paired_resnet.py --seeds 0,1,2     # the same control, end-to-end backbone
for g in sd1.5 midjourney AttnGAN glide xl2.2 DALLE2; do
  python source_only.py --holdout "$g" --out auto   # source-inference control
done
```

`train.py` defaults to the 20-epoch budget used for every ResNet-18 number in the
paper. The four ResNet debiasing variants of Table 13 use the same folds with one
flag added:

```bash
for g in sd1.5 midjourney glide AttnGAN xl2.2 DALLE2; do
  python train.py --protocol logo --holdout "$g" --normalize source   # source-wise target normalisation
  python train.py --protocol logo --holdout "$g" --strong_aug         # heavy augmentation
  python train.py --protocol logo --holdout "$g" --balanced --rank_weight 0.5   # pairwise rank + balanced batches
  python train.py --protocol logo --holdout "$g" --adv 1.0            # adversarial (gradient reversal)
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
python reliability.py                 # attainable ceiling from the per-image rating spread
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
* **paired seen/unseen** — a fixed 20% of each generator is scored twice, once by
  a head trained on the other five generators, once by a head trained on those
  plus the target generator's remaining 80%. The test images are identical in the
  two arms. These test sets are smaller than a LOGO fold, which uses every image
  of the held-out generator, so levels are not directly comparable between the
  two; the paired differences are.
* **logo** — one generator held out entirely. Prompt overlap is *not* removed:
  prompts are shared across generators, so the same prompt can still appear in
  both halves, paired with a different generator's image. Excluding both at once
  would need a joint prompt×generator holdout, which we did not run and do not
  claim.

The `within` / `between` / `overall` figures are diagnostic summaries, not a
variance decomposition: SRCC is a rank statistic, so the overall value is not a
weighted sum of the other two.

The ridge penalty is selected by leave-one-source-out validation inside the
training sources only; the CoOp learning rate is a fixed preset — the held-out generator never participates in
model selection. `nested_ridge.py` prints the α chosen per fold.

---

## Citation

```bibtex
@unpublished{shao2026pooled,
  title  = {The Pooled SRCC Measures Generators: Diagnosing Source Confounding
            in AI-Generated Image Quality Assessment},
  author = {Shao, Yu},
  year   = {2026},
  note   = {Manuscript; code and results at
            https://github.com/AngelBamboo/agiqa-source-confound},
  url    = {https://github.com/AngelBamboo/agiqa-source-confound}
}
```

Journal and volume are filled in if and when the paper is accepted.

## Reporting a model with `report.py`

```bash
python report.py results/predictions/preds_clip_random.csv
python report.py results/predictions/preds_clip_logo_*.csv   # one file per held-out generator
python report.py --json results/predictions/preds_ridge_logo_3k_*.csv
python report.py --pairs results/predictions/preds_clip_random.csv
```

It prints the pooled, within-source and between-source SRCC of each file, and
the equal-weight mean across files — which is the leave-one-generator-out figure
when the files are one per held-out generator. Apply it to any new model's
predictions to see how much of its score comes from ordering sources. 

`--pairs` adds the exact pair decomposition of Kendall's tau: tau over pairs that share a generator, tau over pairs that do not, and the share of the net concordant-minus-discordant count contributed by cross-generator pairs. On the AGIQA-3K random split 75.8% of pairs join two generators, and they carry 83.4% of the net count for every one of the four models tested.

## License

MIT — see `LICENSE`. The datasets keep their own licenses; please cite the
original AGIQA-3K, AIGCIQA2023 and AGIQA-1K papers if you use them.
