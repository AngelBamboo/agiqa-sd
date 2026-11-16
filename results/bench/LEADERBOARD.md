### agiqa1k

| rank (pooled) | model | family | pooled SRCC | within SRCC | between SRCC | LOGO SRCC | rank (within) | shift |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | qualiclip | clip | +0.607 | +0.420 | — | — | 1 | +0 |
| 2 | niqe | classical | +0.569 | +0.253 | — | — | 6 | -4 |
| 3 | brisque | classical | +0.507 | +0.215 | — | — | 9 | -6 |
| 4 | dbcnn | cnn | +0.414 | +0.170 | — | — | 12 | -8 |
| 5 | tres | transformer | +0.411 | +0.132 | — | — | 14 | -9 |
| 6 | musiq | transformer | +0.407 | +0.232 | — | — | 8 | -2 |
| 7 | liqe | clip | +0.398 | +0.288 | — | — | 4 | +3 |
| 8 | clipiqa+ | clip | +0.396 | +0.292 | — | — | 3 | +5 |
| 9 | topiq_nr | transformer | +0.391 | +0.124 | — | — | 15 | -6 |
| 10 | nima | cnn | +0.325 | +0.333 | — | — | 2 | +8 |
| 11 | hyperiqa | cnn | +0.320 | +0.183 | — | — | 10 | +1 |
| 12 | clipiqa | clip | +0.311 | +0.258 | — | — | 5 | +7 |
| 13 | arniqa | transformer | +0.296 | +0.111 | — | — | 16 | -3 |
| 14 | piqe | classical | +0.264 | +0.136 | — | — | 13 | +1 |
| 15 | maniqa | transformer | +0.209 | +0.177 | — | — | 11 | +4 |
| 16 | cnniqa | cnn | +0.147 | +0.233 | — | — | 7 | +9 |

Kendall tau between the two leaderboards: +0.183 (p = 0.35); 49 of 120 model pairs change order.

### agiqa3k

| rank (pooled) | model | family | pooled SRCC | within SRCC | between SRCC | LOGO SRCC | rank (within) | shift |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ridge | linear | +0.863 | +0.563 | +1.000 | +0.579 | 1 | +0 |
| 2 | coop | clip | +0.862 | +0.549 | +1.000 | +0.554 | 2 | +0 |
| 3 | clip_mlp | clip | +0.848 | +0.518 | +1.000 | +0.545 | 4 | -1 |
| 4 | resnet18 | cnn | +0.835 | +0.528 | +1.000 | +0.528 | 3 | +1 |
| 5 | liqe | clip | +0.697 | +0.374 | +0.771 | — | 7 | -2 |
| 6 | clipiqa+ | clip | +0.683 | +0.368 | +0.771 | — | 8 | -2 |
| 7 | topiq_nr | transformer | +0.669 | +0.324 | +0.771 | — | 11 | -4 |
| 8 | qualiclip | clip | +0.661 | +0.456 | +0.771 | — | 5 | +3 |
| 9 | clipiqa | clip | +0.658 | +0.235 | +0.771 | — | 17 | -8 |
| 10 | nima | cnn | +0.657 | +0.222 | +0.943 | — | 19 | -9 |
| 11 | tres | transformer | +0.649 | +0.286 | +0.771 | — | 13 | -2 |
| 12 | dbcnn | cnn | +0.640 | +0.238 | +0.771 | — | 15 | -3 |
| 13 | hyperiqa | cnn | +0.639 | +0.235 | +0.771 | — | 16 | -3 |
| 14 | maniqa | transformer | +0.636 | +0.289 | +0.829 | — | 12 | +2 |
| 15 | musiq | transformer | +0.630 | +0.363 | +0.771 | — | 9 | +6 |
| 16 | arniqa | transformer | +0.618 | +0.230 | +0.771 | — | 18 | -2 |
| 17 | cnniqa | cnn | +0.614 | +0.259 | +0.886 | — | 14 | +3 |
| 18 | niqe | classical | +0.533 | +0.403 | +0.486 | — | 6 | +12 |
| 19 | brisque | classical | +0.497 | +0.341 | +0.543 | — | 10 | +9 |
| 20 | piqe | classical | +0.334 | +0.211 | -0.086 | — | 20 | +0 |

Kendall tau between the two leaderboards: +0.484 (p = 0.00235); 49 of 190 model pairs change order.

Cross-model: pooled~within r = +0.714, pooled~between r = +0.891; OLS pooled = +0.425*within +0.373*between + c (R^2 = 0.920).

### aigciqa2023

| rank (pooled) | model | family | pooled SRCC | within SRCC | between SRCC | LOGO SRCC | rank (within) | shift |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | clip_mlp | clip | +0.813 | +0.441 | +0.943 | +0.431 | 1 | +0 |
| 2 | liqe | clip | +0.767 | +0.384 | +0.943 | — | 2 | +0 |
| 3 | topiq_nr | transformer | +0.751 | +0.331 | +0.943 | — | 5 | -2 |
| 4 | tres | transformer | +0.738 | +0.295 | +0.943 | — | 7 | -3 |
| 5 | musiq | transformer | +0.736 | +0.301 | +0.943 | — | 6 | -1 |
| 6 | arniqa | transformer | +0.728 | +0.274 | +0.943 | — | 9 | -3 |
| 7 | hyperiqa | cnn | +0.726 | +0.281 | +0.943 | — | 8 | -1 |
| 8 | clipiqa+ | clip | +0.716 | +0.342 | +0.943 | — | 4 | +4 |
| 9 | dbcnn | cnn | +0.705 | +0.230 | +0.771 | — | 10 | -1 |
| 10 | maniqa | transformer | +0.670 | +0.227 | +0.943 | — | 11 | -1 |
| 11 | qualiclip | clip | +0.667 | +0.374 | +0.886 | — | 3 | +8 |
| 12 | brisque | classical | +0.624 | +0.160 | +1.000 | — | 14 | -2 |
| 13 | cnniqa | cnn | +0.624 | +0.086 | +0.829 | — | 16 | -3 |
| 14 | clipiqa | clip | +0.609 | +0.212 | +0.943 | — | 13 | +1 |
| 15 | piqe | classical | +0.551 | +0.012 | +1.000 | — | 17 | -2 |
| 16 | nima | cnn | +0.547 | +0.116 | +0.829 | — | 15 | +1 |
| 17 | niqe | classical | +0.506 | +0.224 | +0.543 | — | 12 | +5 |

Kendall tau between the two leaderboards: +0.647 (p = 0.000128); 24 of 136 model pairs change order.

Cross-model: pooled~within r = +0.789, pooled~between r = +0.485; OLS pooled = +0.566*within +0.310*between + c (R^2 = 0.776).

### Cross-dataset stability

| pair | common models | tau (pooled) | tau (within) |
|---|---:|---:|---:|
| agiqa1k vs agiqa3k | 16 | +0.167 | +0.317 |
| agiqa1k vs aigciqa2023 | 16 | +0.033 | +0.050 |
| agiqa3k vs aigciqa2023 | 17 | +0.441 | +0.559 |
