"""Cache frozen CLIP image embeddings for AGIQA-3K and AGIQA-1K.

Extracting once turns every protocol experiment into a head-only fit, which is
what makes the cross-backbone validation affordable.
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import agiqa.config as cfg


class ImageList(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.preprocess(img), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_dir", default=cfg.FEATURES)
    args = ap.parse_args()

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model}_{args.pretrained}".replace("/", "-")

    targets = {
        "agiqa3k": ([r["name"] for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))],
                    os.path.join(cfg.DATA_3K, "images")),
        "agiqa1k": ([r["Image"] for r in csv.DictReader(open(os.path.join(cfg.DATA_1K, "AIGC_all.csv"), encoding="utf-8"))],
                    os.path.join(cfg.DATA_1K, "file")),
    }

    for name, (names, root) in targets.items():
        paths = [os.path.join(root, n) for n in names]
        keep = [i for i, p in enumerate(paths) if os.path.exists(p)]
        paths = [paths[i] for i in keep]
        names = [names[i] for i in keep]
        loader = DataLoader(ImageList(paths, preprocess), batch_size=args.batch, num_workers=0)
        feats = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)
        with torch.no_grad():
            for x, idx in loader:
                f = model.encode_image(x.to(device))
                f = f / f.norm(dim=-1, keepdim=True)
                feats[idx.numpy()] = f.float().cpu().numpy()
        np.save(os.path.join(args.out_dir, f"{name}_{tag}.npy"), feats)
        with open(os.path.join(args.out_dir, f"{name}_{tag}_names.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names))
        print(f"{name}: {feats.shape} saved")


if __name__ == "__main__":
    main()
