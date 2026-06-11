#!/usr/bin/env python3
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a3a5c"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1a3a5c"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=10, leading=14,
                      spaceAfter=6, alignment=4)
FILL = ParagraphStyle("FILL", parent=BODY, textColor=colors.HexColor("#b03a2e"),
                      backColor=colors.HexColor("#fdecea"))
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontSize=17, leading=22)

def fill(txt):
    return Paragraph(f"<b>[FILL AFTER RUNS]</b> {txt}", FILL)

def tbl(header, rows, widths=None):
    data = [header] + rows
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#aab7c4")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5f8")]),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

S = []
S.append(Paragraph("Zero-Shot Fault Detection Across Unseen Industrial Domains", TITLE))
S.append(Paragraph("Domain Gap Analysis Report — AI Internship Screening Task (Deep Learning)", styles["Heading3"]))
S.append(Paragraph("Candidate: ______________ &nbsp;&nbsp;|&nbsp;&nbsp; Date: 12 June 2026 &nbsp;&nbsp;|&nbsp;&nbsp; Code: <i>[GitHub link]</i>", BODY))
S.append(Spacer(1, 8))

S.append(Paragraph("Abstract", H1))
S.append(Paragraph(
    "We address zero-shot anomaly detection on an industrial machine type never observed during "
    "training. A convolutional encoder is trained on normal-only vibration/acoustic recordings of four "
    "MIMII machine types (fan, pump, slider, valve) using masked log-mel spectrogram reconstruction, "
    "while a domain-adversarial head (DANN with gradient reversal) suppresses machine-identity "
    "information in the embedding, yielding a machine-agnostic representation of normal operating "
    "dynamics. At test time, anomalies in the unseen fifth machine type (ToyADMOS ToyCar, a fully "
    "cross-dataset target) are scored by combining the masked-reconstruction error with the "
    "Mahalanobis distance to the pooled source-normal Gaussian — no target-domain samples, labels, "
    "or statistics are used at any stage. We report leave-one-machine-out (LOMO) results within "
    "MIMII, a cross-dataset transfer result, three ablations, and an analysis of performance "
    "degradation as the domain gap widens.", BODY))

S.append(Paragraph("1. Problem Setting and Constraints", H1))
S.append(Paragraph(
    "Training data: normal recordings from 4 source machine types. Test data: normal and anomalous "
    "recordings from a 5th machine type. Hard constraints: (i) no labeled or unlabeled target samples "
    "at training time, which rules out domain adaptation, target-side fine-tuning, and target-fitted "
    "score normalization; (ii) anomaly labels exist only in the test set and are used exclusively for "
    "metric computation. The problem is therefore zero-shot domain generalization for unsupervised "
    "anomaly detection: the model must learn what 'normal industrial machine sound' means in a way "
    "that transfers across machine physics it has never encountered.", BODY))

S.append(Paragraph("2. Datasets and Evaluation Protocol", H1))
S.append(Paragraph(
    "We use the recordings of <b>MIMII</b> (fan, pump, slider, valve; several machine IDs each, "
    "normal/abnormal clips) and <b>ToyADMOS</b> (ToyCar), accessed via the DCASE 2020 Task 2 "
    "development set — the official single-channel 16 kHz repackaging of these two datasets "
    "published by the same research groups. ToyCar provides the genuinely unseen fifth machine type "
    "from a different dataset, recording rig, and acoustic environment. The code also supports the "
    "raw multi-channel MIMII 6 dB/0 dB archives directly. Two evaluation regimes: (a) <b>LOMO "
    "within MIMII</b> — "
    "train on 3 machine types, test zero-shot on the 4th, rotated over all 4 targets (moderate domain "
    "gap, 4 measurement points); (b) <b>cross-dataset</b> — train on all 4 MIMII types, test on ToyCar "
    "(maximal domain gap). Metrics: file-level ROC-AUC and pAUC at FPR ≤ 0.1 (the operating regime "
    "relevant for plant deployment, where false alarms are costly). Patch scores are mean-aggregated "
    "per recording. Seeds fixed (42); all configs serialized with each checkpoint.", BODY))

S.append(Paragraph("3. Method", H1))
S.append(Paragraph("3.1 Features", H2))
S.append(Paragraph(
    "Recordings are resampled to 16 kHz, converted to 128-bin log-mel spectrograms (1024-pt FFT, "
    "512 hop), per-clip standardized, and cut into 64-frame (~2 s) patches with 50% overlap. "
    "Per-clip standardization removes gross level differences between recording rigs — the first, "
    "cheapest line of defense against domain gap.", BODY))
S.append(Paragraph("3.2 Domain-agnostic representation learning", H2))
S.append(Paragraph(
    "A 4-stage CNN encoder maps each patch to a 128-d embedding, trained with two cooperating "
    "objectives. <b>(1) Masked reconstruction:</b> 30% of time columns are masked at the input; a "
    "deconvolutional decoder must reconstruct the full unmasked patch from the embedding. Unlike a "
    "plain autoencoder, this forces the embedding to capture the temporal-spectral regularities of "
    "normal operation (periodicity, harmonic structure, stationarity) rather than memorizing "
    "instances — precisely the properties whose violation defines a fault, independent of machine "
    "type. <b>(2) Domain-adversarial regularization (DANN):</b> a classifier predicts which of the 4 "
    "source machines produced the embedding; a gradient-reversal layer makes the encoder maximize "
    "this classifier's loss. At convergence the embedding is (approximately) invariant to machine "
    "identity, so the source-normal distribution models 'normality' rather than 'fan-ness or "
    "pump-ness'. The reversal coefficient follows the standard warm-up schedule "
    "2/(1+e^(-10p)) - 1 to avoid destabilizing early training.", BODY))
S.append(Paragraph("3.3 Anomaly scoring without target priors", H2))
S.append(Paragraph(
    "After training, a Gaussian (Ledoit-Wolf shrinkage covariance, robust at 128-d) is fitted to "
    "source-normal embeddings only. The test score combines two complementary signals, each "
    "z-normalized using source statistics: the Mahalanobis distance (distributional novelty in the "
    "domain-invariant space) and the masked-reconstruction error (violation of learned normal "
    "dynamics). Mahalanobis excels at global spectral shifts; reconstruction error at transient and "
    "structural irregularities (e.g., valve clicks). The target machine contributes nothing to any "
    "fitted quantity, satisfying the zero-prior constraint by construction.", BODY))
S.append(Paragraph("3.4 Architecture justification", H2))
S.append(tbl(
    ["Design choice", "Justification"],
    [[Paragraph("CNN encoder over transformer", BODY),
      Paragraph("Dataset is small per domain; CNN inductive biases (locality, translation invariance over time-frequency) generalize better at this scale and train within a single GPU session.", BODY)],
     [Paragraph("Masked reconstruction over plain AE", BODY),
      Paragraph("Plain AEs reconstruct anomalies too well (identity shortcut). Masking removes the shortcut and forces predictive modeling of normal dynamics.", BODY)],
     [Paragraph("DANN over CORAL/MMD alignment", BODY),
      Paragraph("Alignment losses match given pairs of domains; DANN learns invariance to the <i>factor</i> of machine identity, which is what must transfer to an unseen 5th domain.", BODY)],
     [Paragraph("Mahalanobis + recon over either alone", BODY),
      Paragraph("Complementary failure modes (global vs. local anomalies); combination is validated in ablation A2.", BODY)],
     [Paragraph("Source-only score normalization", BODY),
      Paragraph("Any target-side normalization would violate the zero-prior constraint; z-statistics are computed from source normals exclusively.", BODY)]],
    widths=[5.2*cm, 11.3*cm]))

S.append(PageBreak())
S.append(Paragraph("4. Results", H1))
S.append(Paragraph("4.1 Main zero-shot results (LOMO + cross-dataset)", H2))
S.append(fill("Insert Table 1 from <i>results/tables.md</i> (generated by "
              "<i>python -m src.make_report_tables</i>). Report AUC and pAUC per unseen target, "
              "plus the MIMII→ToyCar row. Add one sentence comparing the hardest target "
              "(typically valve — non-stationary impulsive sounds) to the easiest (typically fan)."))
S.append(Paragraph("4.2 Ablation A1 — domain-adversarial head", H2))
S.append(fill("Insert Table 2. Expected pattern: adversarial ON improves AUC on unseen targets "
              "(invariance helps transfer) and helps most on the cross-dataset target; if it "
              "slightly hurts an in-dataset target, discuss the invariance-vs-information trade-off."))
S.append(Paragraph("4.3 Ablation A2 — scoring function", H2))
S.append(fill("Insert Table 3 (recon vs. Mahalanobis vs. combined). Comment on which targets favor "
              "which score and whether the combination dominates or merely averages."))
S.append(Paragraph("4.4 Ablation A3 — number of source domains", H2))
S.append(fill("Insert Table 4 (2 vs. 3 vs. 4 source machines, fixed target). Expected: monotone "
              "improvement with source diversity — the core argument that representation "
              "generality, not capacity, drives zero-shot transfer."))

S.append(Paragraph("5. Domain-Gap Degradation Analysis", H1))
S.append(Paragraph(
    "We order test conditions by increasing domain gap: (i) LOMO targets acoustically similar to a "
    "source (e.g., pump unseen, fan in sources — both rotary/stationary); (ii) LOMO targets with "
    "dissimilar physics (valve unseen — impulsive, non-stationary, while sources are mostly "
    "stationary); (iii) cross-dataset ToyCar (different machine class, recording rig, room acoustics, "
    "SNR profile). For quantitative grounding, we also measure the gap directly as the Fréchet "
    "distance between source-normal and target-normal embedding distributions and correlate it with "
    "AUC.", BODY))
S.append(fill("Insert <i>results/domain_gap.png</i> and report: AUC at each gap level, the AUC drop "
              "from the easiest LOMO target to ToyCar, and the correlation between embedding-space "
              "Fréchet distance and AUC. Conclude with where the method degrades gracefully vs. "
              "where it breaks (typically: stationary→stationary transfers hold; "
              "stationary→impulsive and cross-rig transfers lose the most)."))

S.append(Paragraph("6. Limitations and Future Work", H1))
S.append(Paragraph(
    "(1) Invariance is enforced over only four source domains; with so few domains, DANN can remove "
    "useful information along with identity. More source diversity (e.g., adding ToyConveyor/ToyTrain "
    "as sources) is the most direct improvement. (2) Mean patch aggregation dilutes short transients; "
    "top-k aggregation is a cheap fix. (3) The Gaussian assumption on pooled normals is crude — a "
    "per-domain mixture with min-distance scoring is a natural extension. (4) Self-supervised "
    "pretraining on large unlabeled audio (e.g., BEATs/AudioMAE features) would likely raise the "
    "floor, at the cost of compute beyond this task's scope.", BODY))

S.append(Paragraph("7. Reproducibility", H1))
S.append(Paragraph(
    "All code, fixed seeds, and per-run serialized configs are in the repository (README contains "
    "exact commands). The complete ablation grid (~8 short runs) fits one Colab T4 session via "
    "<i>python -m src.run_ablations</i>; results append to a single CSV from which all tables and "
    "the degradation figure are regenerated deterministically.", BODY))

S.append(Paragraph("References", H1))
for r in [
    "Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by Backpropagation (DANN). ICML.",
    "Purohit et al. (2019). MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation. DCASE.",
    "Koizumi et al. (2019). ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection. WASPAA.",
    "Ledoit & Wolf (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. J. Multivariate Analysis.",
    "He et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.",
]:
    S.append(Paragraph(f"• {r}", BODY))

doc = SimpleDocTemplate("/home/claude/zsfd/report/Domain_Gap_Analysis_Report.pdf",
                        pagesize=A4, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        leftMargin=2*cm, rightMargin=2*cm,
                        title="Zero-Shot Fault Detection — Domain Gap Analysis")
doc.build(S)
print("PDF written")
