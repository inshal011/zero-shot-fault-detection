"""Runs the full ablation grid + domain-gap sweep, writes results/results.csv.

Axes:
  A1  adversarial head ON vs OFF
  A2  scoring: recon-only vs Mahalanobis-only vs combined
      (all three modes are computed in every evaluate() call)
  A3  number of source domains: 2 vs 3 vs 4 (representation generality)
  GAP leave-one-machine-out within MIMII (small gap) and
      MIMII -> ToyADMOS ToyCar (large cross-dataset gap)

Designed to fit a single Colab T4 session: ~8 short training runs.
Reduce --epochs if pressed for time; rankings stabilize early.
"""
import argparse

from .train import train, get_parser as train_parser
from .evaluate import evaluate, get_parser as eval_parser

MACHINES = ["fan", "pump", "slider", "valve"]


def run(mimii_root, toy_root, epochs=15, results_csv="results/results.csv"):
    tp, ep = train_parser(), eval_parser()

    def t(**kw):
        a = tp.parse_args(["--mimii-root", mimii_root])
        for k, v in kw.items():
            setattr(a, k.replace("-", "_"), v)
        a.epochs = kw.get("epochs", epochs)
        return train(a)

    def e(**kw):
        a = ep.parse_args(["--mimii-root", mimii_root,
                           "--model", kw.pop("model"),
                           "--target-machine", kw.pop("target_machine")])
        for k, v in kw.items():
            setattr(a, k.replace("-", "_"), v)
        a.results_csv = results_csv
        return evaluate(a)

    # --- GAP + A1: LOMO within MIMII, adversarial on/off -------------------
    for target in MACHINES:
        sources = [m for m in MACHINES if m != target]
        for adv_off in [False, True]:
            tag = f"runs/lomo_{target}{'_noadv' if adv_off else ''}"
            t(source_machines=sources, out=tag, no_adversarial=adv_off)
            e(model=f"{tag}/model.pt", target_machine=target)

    # --- A3: number of source domains, fixed target = valve ----------------
    for k in [2, 3]:
        sources = [m for m in MACHINES if m != "valve"][:k]
        tag = f"runs/nsrc{k}_valve"
        t(source_machines=sources, out=tag)
        e(model=f"{tag}/model.pt", target_machine="valve")

    # --- GAP: cross-dataset, all 4 MIMII -> ToyADMOS ToyCar ----------------
    if toy_root:
        tag = "runs/all4"
        t(source_machines=MACHINES, out=tag)
        e(model=f"{tag}/model.pt", target_machine="ToyCar",
          target_root=toy_root)

    print(f"\nAll ablations done. Results -> {results_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mimii-root", required=True)
    p.add_argument("--toy-root", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--results-csv", default="results/results.csv")
    a = p.parse_args()
    run(a.mimii_root, a.toy_root, a.epochs, a.results_csv)
