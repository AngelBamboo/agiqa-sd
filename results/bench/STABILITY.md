# Per-generator leaderboard stability

## agiqa1k

Pairwise Kendall tau between generator rankings: mean -0.050, range -0.050 to -0.050.
Distinct winners across 2 generators: 2.

| generator | best model |
|---|---|
| deepai | nima |
| dreamStudio | qualiclip |

| | deepai | dreamStudio |
|---|---:|---:|
| **deepai** | +1.00 | -0.05 |
| **dreamStudio** | -0.05 | +1.00 |

## agiqa3k

Pairwise Kendall tau between generator rankings: mean +0.425, range +0.158 to +0.632.
Distinct winners across 6 generators: 4.

| generator | best model |
|---|---|
| AttnGAN | niqe |
| DALLE2 | coop |
| glide | resnet18 |
| midjourney | ridge |
| sd1.5 | coop |
| xl2.2 | coop |

| | AttnGAN | DALLE2 | glide | midjourney | sd1.5 | xl2.2 |
|---|---:|---:|---:|---:|---:|---:|
| **AttnGAN** | +1.00 | +0.38 | +0.42 | +0.34 | +0.16 | +0.38 |
| **DALLE2** | +0.38 | +1.00 | +0.45 | +0.54 | +0.55 | +0.52 |
| **glide** | +0.42 | +0.45 | +1.00 | +0.41 | +0.44 | +0.41 |
| **midjourney** | +0.34 | +0.54 | +0.41 | +1.00 | +0.63 | +0.41 |
| **sd1.5** | +0.16 | +0.55 | +0.44 | +0.63 | +1.00 | +0.34 |
| **xl2.2** | +0.38 | +0.52 | +0.41 | +0.41 | +0.34 | +1.00 |

## aigciqa2023

Pairwise Kendall tau between generator rankings: mean +0.328, range -0.162 to +0.765.
Distinct winners across 6 generators: 4.

| generator | best model |
|---|---|
| Controlnet | clip_mlp |
| DALLE | qualiclip |
| Glide | clip_mlp |
| Lafite | tres |
| Unidiffuser | clipiqa+ |
| stable-diffusion | clip_mlp |

| | Controlnet | DALLE | Glide | Lafite | Unidiffuser | stable-diffusion |
|---|---:|---:|---:|---:|---:|---:|
| **Controlnet** | +1.00 | +0.44 | +0.44 | +0.06 | +0.76 | +0.69 |
| **DALLE** | +0.44 | +1.00 | +0.24 | +0.15 | +0.47 | +0.37 |
| **Glide** | +0.44 | +0.24 | +1.00 | +0.29 | +0.32 | +0.19 |
| **Lafite** | +0.06 | +0.15 | +0.29 | +1.00 | +0.09 | -0.16 |
| **Unidiffuser** | +0.76 | +0.47 | +0.32 | +0.09 | +1.00 | +0.57 |
| **stable-diffusion** | +0.69 | +0.37 | +0.19 | -0.16 | +0.57 | +1.00 |
