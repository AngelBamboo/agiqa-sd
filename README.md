# AGIQA-SD: source-disjoint benchmarking for AI-generated image quality assessment

Most AGIQA papers report one number: the pooled Spearman rank correlation between
predicted and mean opinion scores on a random split. A random split places every
generator in both training and test sets, so a model can reach a high pooled
correlation by recognising which generator produced an image.

This repository is the benchmark that separates the two abilities. It reports
three correlations, on three public databases and up to 20 models per database,
with fixed splits, per-image scores and one-command reproduction.

| Axis | What it measures |
|---|---|
| **Pooled SRCC** | ranking over the whole test set, the number almost all papers report |
| **Within-generator SRCC** | ranking images inside each generator, averaged with equal weight |
| **Between-generator SRCC** | ordering the generator mean scores |

## Headline results

| Database | Models | Kendall tau (pooled vs within) | Flipped pairs |
|---|---:|---:|---:|
| AGIQA-3K | 20 | 0.484 | 49 / 190 |
| AIGCIQA2023 | 17 | 0.647 | 24 / 136 |
| AGIQA-1K | 16 | 0.183 | 49 / 120 |

Cross-database rank agreement falls to Kendall tau = 0.033 between AGIQA-1K and
AIGCIQA2023, and four different models are best on the six generators of each
multi-generator database. Pooled SRCC is explained by its within and between
components with R-squared = 0.920 on AGIQA-3K.

## Repository layout

```
bench/                  the benchmark harness
  datasets.py           unified loaders for the three databases
  registry.py           model registry (16 training-free metrics)
  runner.py             scoring and the three-axis decomposition
  trained.py            folds the trained baselines' predictions into records
  leaderboard.py        rankings, rank correlation, cross-model regression
  stability.py          per-generator rank stability
  bootstrap.py          stratified bootstrap intervals
  figures.py            the five paper figures
  export_tables.py      CSV and Markdown tables for the supplement
  export_scores.py      per-image scores as CSV
agiqa/                  legacy loaders and training code for the four baselines
scripts/                model download and overnight matrix runner
results/
  bench/summary/        one JSON record per model and database
  bench/scores/         per-image score arrays (.npy, local cache; released copy is scores_csv/)
  bench/scores_csv/     the same scores as CSV
  bench/bootstrap/      interval records
  bench/tables/         released tables
  predictions/          per-image predictions of the trained baselines
figs/bench/             generated figures (PDF and PNG)
```

## Quick start

```bash
conda create -n agiqa python=3.10
conda activate agiqa
pip install -r requirements.txt

export AGIQA_3K=/path/to/agiqa3k        # contains data.csv and images/
export AGIQA_2023=/path/to/aigciqa2023  # contains data.csv and allimg/
export AGIQA_1K=/path/to/agiqa1k        # contains AIGC_all.csv and file/
```

On Windows, set the same names as environment variables.

## Data

The images are not redistributed here. Download each database from its release
and point the environment variables above at the unpacked folders.

| Database | Link |
|---|---|
| AGIQA-3K | https://github.com/lcysyzxdxc/AGIQA-3K |
| AIGCIQA2023 | https://github.com/wanghaining/2023-AIGC |
| AGIQA-1K | https://github.com/lcysyzxdxc/AGIQA-1K |

Each database keeps its own license and terms of use; cite the corresponding
paper if you use it.

## Reproduce the results

```bash
# 1. training-free metrics, all three databases (downloads weights on first run)
python scripts/download_models.py
for ds in agiqa3k aigciqa2023 agiqa1k; do
  python -m bench.runner --dataset $ds --models all
done

# 2. trained baselines from the archived per-image predictions
python -m bench.trained --dataset agiqa3k
python -m bench.trained --dataset aigciqa2023

# 3. analyses
python -m bench.leaderboard          # rankings, tau, flipped pairs, OLS
python -m bench.stability            # per-generator rank stability
python -m bench.bootstrap --dataset agiqa3k --reps 200

# 4. tables and figures
python -m bench.export_tables
python -m bench.export_scores
python -m bench.figures
```

Every runner skips work already on disk, so an interrupted matrix resumes where
it stopped. `scripts/overnight.sh` chains the download and the three databases.

## Record format

Each file in `results/bench/summary` is a JSON object with:

```
dataset, model, family, n, n_generators,
pooled {srcc, plcc, krcc},
within_mean, within_weighted, between,
per_generator {<generator>: {n, srcc, plcc, krcc, mos_mean, score_mean}},
meta {lower_better, n_scored, n_failed, seconds}
```

Trained records add a `random` block (the same decomposition on the random-split
test set) and a `logo` block (the mean of the six leave-one-generator-out
correlations), plus a `source_disjoint` alias for the leaderboard.

## Notes

- IL-NIQE is excluded: it runs at about one image per second and its SVD fails on
  a handful of generated images, which poisons the CUDA context.
- TReS weights are downloaded from `hf-mirror.com` when `huggingface.co` is
  unreachable; see `scripts/download_models.py`.
- The earlier diagnostic paper, *The Pooled SRCC Measures Generators*, is
  superseded by this benchmark. Its records remain under `results/runs` and its
  manuscript under `archive/`.

## Citation

```bibtex
@article{shao2026agiqasd,
  title   = {Source-Disjoint Benchmarking for AI-Generated Image Quality Assessment},
  author  = {Shao, Yu},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License. The databases keep their own licenses.
