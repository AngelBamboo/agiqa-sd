# Data

The images are not redistributed in this repository. Download each database from
its original source and unpack it here (or point the environment variables in
`agiqa/config.py` at wherever you keep it).

| Folder | Dataset | Download |
|---|---|---|
| `agiqa3k/` | AGIQA-3K | https://github.com/lcysyzxdxc/AGIQA-3K |
| `aigciqa2023/` | AIGCIQA2023 | release linked from https://github.com/wanghaining/2023-AIGC |
| `agiqa1k/` | AGIQA-1K | https://github.com/lcysyzxdxc/AGIQA-1K |

Layout expected by `agiqa/data.py`:

```
data/agiqa3k/        data.csv   images/       # one row per image: name, mos, std, generator
data/aigciqa2023/    data.csv   allimg/       # plus the JSON index with the five rater scores
data/agiqa1k/        AIGC_all.csv   file/
```

If a folder is laid out differently, adapt the reader in `agiqa/data.py` — each
dataset has its own small loader and nothing else depends on the layout.

Cached CLIP features go to `features/` (created automatically by
`extract_features.py`); each file is a single `.npy` array aligned with the row
order of the dataset's CSV.

License note: the three databases keep their own licenses and terms of use. Cite
the corresponding papers if you use them.
