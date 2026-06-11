"""Turns results/results.csv into report-ready tables + degradation plot.

    python -m src.make_report_tables --results results/results.csv

Outputs:
    results/tables.md          (paste into the report)
    results/domain_gap.png     (AUC vs. domain-gap condition)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(results_csv):
    df = pd.read_csv(results_csv)
    out = []

    # Table 1: main LOMO results (combined score, adversarial on)
    t1 = df[(df.score_mode == "combined") & (df.adversarial) &
            (df.n_sources == 3)]
    out.append("### Table 1 — Zero-shot LOMO results (combined score)\n")
    out.append(t1[["target", "sources", "auc", "pauc_0.1"]]
               .to_markdown(index=False))

    # Table 2: A1 adversarial ablation
    t2 = (df[df.score_mode == "combined"]
          .pivot_table(index="target", columns="adversarial", values="auc"))
    out.append("\n\n### Table 2 — Effect of domain-adversarial head (AUC)\n")
    out.append(t2.to_markdown())

    # Table 3: A2 scoring ablation
    t3 = (df[df.adversarial]
          .pivot_table(index="target", columns="score_mode", values="auc"))
    out.append("\n\n### Table 3 — Scoring-function ablation (AUC)\n")
    out.append(t3.to_markdown())

    # Table 4: A3 number of source domains
    t4 = (df[(df.target == "valve") & (df.score_mode == "combined")]
          .sort_values("n_sources")[["n_sources", "sources", "auc",
                                     "pauc_0.1"]])
    out.append("\n\n### Table 4 — Source-domain count vs. generalization\n")
    out.append(t4.to_markdown(index=False))

    with open("results/tables.md", "w") as f:
        f.write("\n".join(out))
    print("wrote results/tables.md")

    # Degradation plot: in-dataset LOMO targets vs cross-dataset target
    plot_df = df[(df.score_mode == "combined") & (df.adversarial)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(plot_df.target, plot_df.auc, color="#4878a8")
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set_ylabel("File-level AUC")
    ax.set_title("Zero-shot AUC by unseen target (left: LOMO, right: cross-dataset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/domain_gap.png", dpi=160)
    print("wrote results/domain_gap.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.csv")
    main(p.parse_args().results)
