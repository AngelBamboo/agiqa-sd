"""Data loading and split construction for AGIQA-3K.

Three protocols are supported because the paper's central claim is about how
the evaluation protocol changes the apparent quality of a model:

    random   : stratified random 80/20, the setting most papers report
    prompt   : grouped by prompt text, so the same prompt never spans the split
    logo     : leave-one-generator-out, the only source-disjoint protocol
"""
import csv
import hashlib
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GENERATORS = ["sd1.5", "midjourney", "AttnGAN", "glide", "xl2.2", "DALLE2"]


def load_metadata(root):
    """Returns a list of dicts with image path, MOS, prompt and generator."""
    csv_path = os.path.join(root, "data.csv")
    img_dir = os.path.join(root, "images")
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["name"]
            path = os.path.join(img_dir, name)
            if not os.path.exists(path):
                continue
            rows.append({
                "path": path,
                "name": name,
                "prompt": r["prompt"].strip(),
                "generator": name.split("_")[0],
                "mos": float(r["mos_quality"]),
                "mos_align": float(r["mos_align"]),
            })
    return rows


def _prompt_key(prompt):
    return hashlib.md5(prompt.lower().encode("utf-8")).hexdigest()


def make_split(rows, protocol, seed=0, holdout=None, test_ratio=0.2):
    """Returns (train_idx, test_idx) as integer index arrays."""
    rng = np.random.RandomState(seed)
    n = len(rows)

    if protocol == "random":
        idx = rng.permutation(n)
        cut = int(n * (1 - test_ratio))
        return np.sort(idx[:cut]), np.sort(idx[cut:])

    if protocol == "prompt":
        keys = sorted({_prompt_key(r["prompt"]) for r in rows})
        rng.shuffle(keys)
        cut = int(len(keys) * (1 - test_ratio))
        test_keys = set(keys[cut:])
        test = [i for i, r in enumerate(rows) if _prompt_key(r["prompt"]) in test_keys]
        train = [i for i, r in enumerate(rows) if _prompt_key(r["prompt"]) not in test_keys]
        return np.array(train), np.array(test)

    if protocol == "logo":
        if holdout not in GENERATORS:
            raise ValueError(f"holdout must be one of {GENERATORS}")
        train = [i for i, r in enumerate(rows) if r["generator"] != holdout]
        test = [i for i, r in enumerate(rows) if r["generator"] == holdout]
        return np.array(train), np.array(test)

    raise ValueError(f"unknown protocol {protocol}")


class AGIQADataset(Dataset):
    """In-memory uint8 cache keeps the small dataset fully on the host."""

    def __init__(self, rows, indices, size=224, train=False, mos_key="mos",
                 return_generator=False, normalize="global", strong_aug=False, local=False):
        self.rows = [rows[i] for i in indices]
        self.size = size
        self.train = train
        self.mos_key = mos_key
        self.return_generator = return_generator
        self.strong_aug = strong_aug
        self.local = local
        self.cache = [self._read(r["path"]) for r in self.rows]

        mos = np.array([r[mos_key] for r in self.rows], dtype=np.float32)
        self.mos_mean, self.mos_std = float(mos.mean()), float(mos.std() + 1e-8)

        # Source-wise normalisation: the regression target becomes "quality
        # relative to other images from the same generator", which removes the
        # between-generator offset a model could otherwise exploit.
        self.normalize = normalize
        self.source_stats = {}
        if normalize == "source":
            for g in {r["generator"] for r in self.rows}:
                vals = np.array([r[mos_key] for r in self.rows if r["generator"] == g],
                                dtype=np.float32)
                self.source_stats[g] = (float(vals.mean()), float(vals.std() + 1e-8))

    def _read(self, path):
        img = Image.open(path).convert("RGB")
        if self.train:
            # random-resized crop, keeps aspect variability without extra deps
            w, h = img.size
            lo, hi = (0.25, 0.6) if self.local else (0.7, 1.0)
            scale = np.random.uniform(lo, hi)
            cw, ch = int(w * scale), int(h * scale)
            x0 = np.random.randint(0, max(1, w - cw + 1))
            y0 = np.random.randint(0, max(1, h - ch + 1))
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
        img = img.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.uint8)
        if self.train and self.strong_aug:
            arr = self._strong_aug(arr)
        return arr

    def _strong_aug(self, arr):
        """Perturbs cues that are specific to a generator's decoder.

        Generator fingerprints live in fine high-frequency statistics (upsampling
        kernels, JPEG quantisation, noise spectra). Re-encoding with a random
        quality, resampling, and adding mild noise forces the network towards
        distortion cues that every generator shares.
        """
        import cv2
        if np.random.rand() < 0.8:
            q = np.random.randint(30, 96)
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                arr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if np.random.rand() < 0.5:
            s = np.random.uniform(0.6, 0.95)
            h, w = arr.shape[:2]
            small = cv2.resize(arr, (max(8, int(w * s)), max(8, int(h * s))),
                               interpolation=cv2.INTER_AREA)
            arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if np.random.rand() < 0.4:
            sigma = np.random.uniform(0, 6)
            arr = np.clip(arr.astype(np.float32) +
                          np.random.randn(*arr.shape) * sigma, 0, 255).astype(np.uint8)
        return arr

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr = self.cache[i].astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1]
        arr = (arr - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        if self.normalize == "source":
            mu, sd = self.source_stats[self.rows[i]["generator"]]
        else:
            mu, sd = self.mos_mean, self.mos_std
        y = torch.tensor((self.rows[i][self.mos_key] - mu) / sd, dtype=torch.float32)
        if self.return_generator:
            return x, y, GENERATORS.index(self.rows[i]["generator"])
        return x, y
