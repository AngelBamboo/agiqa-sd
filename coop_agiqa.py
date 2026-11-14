import os
"""CoOp-style prompt learning for AGIQA, used as a published-method baseline.

The image tower stays frozen; only the prompt context, a temperature and an
optional linear readout are trained. This mirrors the CLIP-AGIQA recipe [4],
which builds on CoOp [5], so that the source-confound analysis can be repeated
on a prompt-learning model rather than only on a plain MLP head.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agiqa.data import make_split
import agiqa.config as cfg

LEVELS = ["very low", "low", "medium", "high", "very high"]
LEVEL_VALUES = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])


class CoOpQuality(nn.Module):
    def __init__(self, clip_model, n_ctx=16, init_text="a photo of a"):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad_(False)

        import open_clip
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.n_ctx = n_ctx
        self.register_buffer("level_values", LEVEL_VALUES)

        # CoOp sequence layout: [SOT] + n_ctx learnable tokens + class tokens + [EOT]
        classes = [f"{lv} quality image" for lv in LEVELS]
        raw = self.tokenizer(classes)                       # (K, 77)
        sot, eot = raw[0, 0].item(), raw.max(dim=1).values
        rows = []
        for i, cls in enumerate(classes):
            n_cls = int((raw[i] == eot[i]).nonzero()[0]) - 1
            class_ids = raw[i, 1:1 + n_cls]
            seq = torch.cat([
                raw[i, :1],
                torch.zeros(n_ctx, dtype=torch.long),
                class_ids,
                torch.tensor([eot[i]],
                             dtype=torch.long),
            ])
            seq = torch.cat([seq, torch.zeros(77 - len(seq), dtype=torch.long)])
            rows.append(seq)
        ids = torch.stack(rows)
        self.register_buffer("ids", ids)
        with torch.no_grad():
            init_ids = self.tokenizer([f"{init_text} {lv} quality image" for lv in LEVELS])
            init_ids = init_ids.to(self.clip.token_embedding.weight.device)
            emb = self.clip.token_embedding(init_ids)
            ctx_init = emb[:, 1:1 + n_ctx, :].mean(0, keepdim=True)
        self.ctx = nn.Parameter(ctx_init.clone())
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.readout = nn.Linear(5, 1) if False else None

    def text_features(self):
        x = self.clip.token_embedding(self.ids)
        ctx = self.ctx.expand(x.shape[0], -1, -1)
        x = torch.cat([x[:, :1], ctx, x[:, 1 + self.n_ctx:]], dim=1)
        x = x + self.clip.positional_embedding
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = self.clip.ln_final(x)
        from open_clip.transformer import text_global_pool
        x = text_global_pool(x, self.ids, self.clip.text_pool_type,
                             eos_token_id=getattr(self.clip, "text_eos_id", None))
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                x = self.clip.text_projection(x)
            else:
                x = x @ self.clip.text_projection
        return F.normalize(x, dim=-1)

    def forward(self, image_feats):
        t = self.text_features()
        logits = self.logit_scale.exp() * image_feats @ t.t()
        prob = logits.softmax(-1)
        return prob @ self.level_values.to(prob.device)


def load_feats():
    names = open(os.path.join(cfg.FEATURES, "agiqa3k_ViT-B-32_openai_names.txt"),
                 encoding="utf-8").read().split("\n")
    X = torch.from_numpy(np.load(cfg.CLIP_3K)).float()
    meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(cfg.DATA_3K, "data.csv"), encoding="utf-8"))}
    rows = [{"name": n, "generator": n.split("_")[0],
             "mos": float(meta[n]["mos_quality"])} for n in names]
    return X, rows


def run(protocol, holdout, seed, epochs, lr, dump=""):
    torch.manual_seed(seed)
    X, rows = load_feats()
    y = torch.tensor([r["mos"] for r in rows], dtype=torch.float32)
    gens = [r["generator"] for r in rows]
    tr, te = make_split(rows, protocol, seed=seed, holdout=holdout)

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.cuda().eval()
    model = CoOpQuality(clip_model).cuda()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    Xtr, ytr = X[tr].cuda(), y[tr].cuda()
    n = len(tr)
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(256, n),))
        pred = model(Xtr[idx])
        loss = F.mse_loss(pred, ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = model(X[te].cuda()).cpu().numpy()
    gt = y[te].numpy()
    if dump:
        import csv as _csv
        with open(dump, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["name", "generator", "mos", "pred"])
            for k, i in enumerate(te):
                w.writerow([rows[i]["name"], rows[i]["generator"], rows[i]["mos"], float(pred[k])])
    per = {}
    gte = [gens[i] for i in te]
    for g in set(gte):
        sel = np.array(gte) == g
        if sel.sum() > 30:
            per[g] = float(stats.spearmanr(pred[sel], gt[sel]).statistic)
    return {"protocol": protocol, "holdout": holdout if protocol == "logo" else None,
            "seed": seed,
            "srcc": float(stats.spearmanr(pred, gt).statistic),
            "plcc": float(stats.pearsonr(pred, gt).statistic),
            "within": float(np.mean(list(per.values()))) if per else None,
            "per_generator": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="random,logo")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", default=os.path.join(cfg.RUNS, "coop_results.json"))
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for proto in args.protocols.split(","):
        holds = [None] if proto == "random" else ["sd1.5", "AttnGAN", "glide", "xl2.2", "DALLE2", "midjourney"]
        for h in holds:
            for s in seeds:
                dump = args.dump if (args.dump and proto == "random" and s == 0) else ""
                r = run(proto, h, s, args.epochs, args.lr, dump)
                results.append(r)
                print(f"{proto:7s} {str(h or '-'):10s} s{s} srcc {r['srcc']:.3f}"
                      f" within {r['within'] if r['within'] is None else round(r['within'],3)}", flush=True)

    rand = [r["srcc"] for r in results if r["protocol"] == "random"]
    logo = [r["srcc"] for r in results if r["protocol"] == "logo"]
    print(f"\nCoOp random {np.mean(rand):.3f} ± {np.std(rand):.3f}")
    print(f"CoOp LOGO   {np.mean(logo):.3f} ± {np.std(logo):.3f}")
    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
