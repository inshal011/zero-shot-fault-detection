"""Zero-shot evaluation on an unseen machine type.

Protocol:
  1. Load a model trained on N source machines (target machine excluded).
  2. Fit scoring statistics on SOURCE normal embeddings only.
  3. Score the unseen target machine's normal + anomalous recordings.
  4. Report file-level AUC and pAUC (FPR <= 0.1).

Usage:
    python -m src.evaluate --model runs/no_valve/model.pt \
        --mimii-root data/mimii_6dB --target-machine valve \
        --results-csv results/results.csv

Cross-dataset zero-shot (the actual task setting — 5th machine type
from a different dataset entirely):
    python -m src.evaluate --model runs/all4/model.pt \
        --mimii-root data/mimii_6dB \
        --target-root data/toyadmos --target-machine ToyCar \
        --results-csv results/results.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import build_source_train, build_target_test
from .models import ZeroShotModel
from .score import SourceNormalScorer, embed_dataset, file_level_scores


def pauc(y, s, max_fpr=0.1):
    return roc_auc_score(y, s, max_fpr=max_fpr)


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["args"]
    model = ZeroShotModel(
        n_mels=cfg["n_mels"], n_frames=cfg["n_frames"], emb_dim=cfg["emb_dim"],
        n_domains=len(cfg["source_machines"]),
        use_adversarial=not cfg["no_adversarial"]).to(device)
    model.load_state_dict(ckpt["model"])

    # 1) source normals -> scorer statistics (no target data involved)
    src = build_source_train(args.mimii_root, cfg["source_machines"],
                             n_frames=cfg["n_frames"],
                             patches_per_file=cfg["patches_per_file"])
    z_src, err_src, _, _ = embed_dataset(model, src, device)
    scorer = SourceNormalScorer().fit(z_src, err_src)

    # 2) unseen target machine
    target_root = args.target_root or args.mimii_root
    tgt = build_target_test(target_root, args.target_machine,
                            n_frames=cfg["n_frames"],
                            max_files_per_class=args.max_files_per_class)
    z_t, err_t, y_t, fid_t = embed_dataset(model, tgt, device)

    rows = []
    for mode in ["recon", "maha", "combined"]:
        ps = scorer.score(z_t, err_t, mode=mode)
        fs, fy = file_level_scores(ps, y_t, fid_t)
        auc = roc_auc_score(fy, fs)
        pa = pauc(fy, fs)
        rows.append({
            "target": args.target_machine,
            "sources": "+".join(cfg["source_machines"]),
            "n_sources": len(cfg["source_machines"]),
            "adversarial": not cfg["no_adversarial"],
            "mask_ratio": cfg["mask_ratio"],
            "score_mode": mode,
            "auc": round(float(auc), 4),
            "pauc_0.1": round(float(pa), 4),
            "n_test_files": int(len(fy)),
        })
        print(f"[{args.target_machine}] {mode:9s} AUC={auc:.4f} pAUC={pa:.4f}")

    if args.results_csv:
        out = Path(args.results_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"appended {len(rows)} rows -> {out}")
    return rows


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mimii-root", required=True,
                   help="root of source (training) dataset")
    p.add_argument("--target-root", default=None,
                   help="root of target dataset if different (e.g. ToyADMOS)")
    p.add_argument("--target-machine", required=True)
    p.add_argument("--max-files-per-class", type=int, default=None)
    p.add_argument("--results-csv", default="results/results.csv")
    return p


if __name__ == "__main__":
    evaluate(get_parser().parse_args())
