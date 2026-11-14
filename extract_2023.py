"""Cache CLIP features for AIGCIQA2023."""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import agiqa.config as cfg


class Imgs(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert("RGB")), i


def main():
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    rows = list(csv.DictReader(open(os.path.join(cfg.DATA_2023, "data.csv"), encoding="utf-8")))
    paths = [os.path.join(os.path.join(cfg.DATA_2023, "allimg"), r["name"]) for r in rows]
    loader = DataLoader(Imgs(paths, preprocess), batch_size=64, num_workers=0)
    feats = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)
    with torch.no_grad():
        for x, idx in loader:
            f = model.encode_image(x.to(device))
            f = f / f.norm(dim=-1, keepdim=True)
            feats[idx.numpy()] = f.float().cpu().numpy()
    os.makedirs(cfg.FEATURES, exist_ok=True)
    np.save(cfg.CLIP_2023, feats)
    print("saved", feats.shape)


if __name__ == "__main__":
    main()
