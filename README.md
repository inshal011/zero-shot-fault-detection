# Zero-Shot Fault Detection Across Unseen Industrial Domains

Train on vibration/acoustic recordings from 4 machine types (fan, pump,
slider, valve — MIMII) and detect anomalies on a **5th machine type never
seen during training** (cross-dataset: ToyADMOS ToyCar), with **no target
domain samples, labels, or statistics used at any point**.

## Method (summary)

1. **Features** — 128-bin log-mel spectrograms, cut into 64-frame patches
   (~2 s), per-clip standardized.
2. **Domain-agnostic representation** — a CNN encoder trained with
   **masked spectrogram reconstruction** (self-supervised; learns the
   dynamics of "normal" machine sound) **plus a domain-adversarial head**
   (DANN, gradient reversal): the encoder is penalized whenever the
   embedding reveals *which* machine type produced the sound, forcing it
   to encode machine-independent normality structure.
3. **Anomaly scoring without target priors** — fit a Ledoit-Wolf Gaussian
   on **source-normal embeddings only**; score = z-normalized
   **Mahalanobis distance + masked-reconstruction error**. The target
   machine contributes nothing to any fitted statistic.
4. **Evaluation** — leave-one-machine-out (LOMO) within MIMII for the
   domain-gap sweep, plus MIMII→ToyADMOS for the full cross-dataset
   zero-shot setting. File-level AUC and pAUC (FPR ≤ 0.1).

## Setup

```bash
pip install -r requirements.txt
```

### Datasets

| Dataset | Link | Used subset |
|---|---|---|
| MIMII | https://zenodo.org/record/3384388 | 6 dB SNR, all 4 machines (0 dB also supported) |
| ToyADMOS | https://zenodo.org/record/3351307 | ToyCar (IND channel 1 is sufficient) |

Extract so that paths look like:

```
data/mimii_6dB/fan/id_00/normal/*.wav
data/mimii_6dB/pump/id_00/abnormal/*.wav
data/toyadmos/ToyCar/case1/NormalSound_IND/*.wav
data/toyadmos/ToyCar/case1/AnomalousSound_IND/*.wav
```

(The scanners match on path keywords, so minor layout differences are fine.
To save time/disk you may use a subset of `id_*` folders — keep at least
two ids per machine.)

## Reproduce

```bash
# 1. Single LOMO run (train on fan+pump+slider, test zero-shot on valve)
python -m src.train --mimii-root data/mimii_6dB \
    --source-machines fan pump slider --epochs 30 --out runs/no_valve
python -m src.evaluate --model runs/no_valve/model.pt \
    --mimii-root data/mimii_6dB --target-machine valve

# 2. Full ablation grid + domain-gap sweep (≈ one Colab T4 session)
python -m src.run_ablations --mimii-root data/mimii_6dB \
    --toy-root data/toyadmos --epochs 15

# 3. Report tables + degradation plot
python -m src.make_report_tables --results results/results.csv
```

Seeds are fixed (`--seed 42`); configs are dumped to `runs/*/config.json`.

## Ablations covered

| ID | Axis | How |
|---|---|---|
| A1 | Domain-adversarial head on/off | `--no-adversarial` |
| A2 | Score: recon / Mahalanobis / combined | computed in every eval |
| A3 | 2 vs 3 vs 4 source domains | `--source-machines` |
| GAP | Domain-gap degradation | LOMO (small gap) → cross-dataset ToyADMOS (large gap) |

## Repo layout

```
src/
  data.py                MIMII / ToyADMOS scanning, log-mel patch dataset
  models.py              encoder, GRL, domain head, masked decoder
  train.py               training loop (+ ablation flags)
  score.py               source-only Mahalanobis + recon scoring
  evaluate.py            zero-shot eval, AUC/pAUC, results CSV
  run_ablations.py       full grid
  make_report_tables.py  CSV -> report tables + plot
```
