"""Training: masked reconstruction + (optional) domain-adversarial objective.

Usage (example, leave-valve-out):
    python -m src.train --mimii-root data/mimii_6dB \
        --source-machines fan pump slider --epochs 30 --out runs/no_valve

Ablation flags:
    --no-adversarial          disable the DANN head
    --mask-ratio 0.0          disable masking (plain autoencoder)
    --source-machines ...     vary the number of source domains
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import build_source_train
from .models import ZeroShotModel


def grl_schedule(step, total_steps, max_lambda=1.0):
    """DANN lambda warm-up: 2/(1+exp(-10p)) - 1, p = progress in [0,1]."""
    p = step / max(1, total_steps)
    return max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = build_source_train(args.mimii_root, args.source_machines,
                            n_frames=args.n_frames,
                            patches_per_file=args.patches_per_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    model = ZeroShotModel(n_mels=args.n_mels, n_frames=args.n_frames,
                          emb_dim=args.emb_dim,
                          n_domains=len(args.source_machines),
                          use_adversarial=not args.no_adversarial).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    step = 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    for epoch in range(args.epochs):
        model.train()
        tot_rec, tot_dom, n = 0.0, 0.0, 0
        for x, dom, _, _ in loader:
            x, dom = x.to(device), dom.to(device)
            lam = grl_schedule(step, total_steps, args.grl_lambda)
            out = model(x, mask_ratio=args.mask_ratio, grl_lambda=lam)
            loss_rec = F.mse_loss(out["recon"], x)
            loss = loss_rec
            loss_dom = torch.tensor(0.0)
            if not args.no_adversarial:
                loss_dom = F.cross_entropy(out["domain_logits"], dom)
                loss = loss + args.adv_weight * loss_dom
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_rec += loss_rec.item() * x.size(0)
            tot_dom += float(loss_dom) * x.size(0)
            n += x.size(0)
            step += 1
        print(f"epoch {epoch+1}/{args.epochs} "
              f"recon={tot_rec/n:.4f} domain={tot_dom/n:.4f} lambda={lam:.2f}")

    torch.save({"model": model.state_dict(), "args": vars(args)},
               out_dir / "model.pt")
    print(f"saved -> {out_dir/'model.pt'}")
    return model, ds


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mimii-root", required=True)
    p.add_argument("--source-machines", nargs="+",
                   default=["fan", "pump", "slider", "valve"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--grl-lambda", type=float, default=1.0)
    p.add_argument("--adv-weight", type=float, default=0.5)
    p.add_argument("--patches-per-file", type=int, default=8)
    p.add_argument("--no-adversarial", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/exp")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
