#!/usr/bin/env python3
"""
validate_antigen_classifier_score.py
===========================
Run AFTER validate_antigen_classifier_sample.py's output CSV has been hand-labeled
(the `correct_class` column filled in for every wrong row).

Produces, using only the `stratum="random"` rows for anything claiming to
be population-weighted (the only stratum drawn unconditioned on the
predicted class, hence the only one an unbiased accuracy estimate can
legitimately come from):

  - Overall accuracy (prevalence-weighted, from the random stratum alone),
    with a Wilson-score 95% CI.
  - A full confusion matrix (predicted class x ground-truth class),
    combining all three strata (valid for building the matrix itself,
    since each cell is just a count -- NOT valid for reading off a
    population-weighted rate directly from combined-stratum cell counts,
    which is why accuracy is computed separately, above, from the random
    stratum only).
  - Per-class precision, recall, and F1, computed from the random +
    rare_class_oversample strata combined (valid for precision, which is
    conditioned on the predicted class and therefore not biased by how
    classes were sampled; recall from combined strata is reported but
    flagged, since oversampling changes the effective denominator for
    classes that were rarely predicted -- true recall for very rare TRUE
    classes cannot be fully recovered without knowing how often the
    classifier fails to predict them at all, which no finite audit of
    already-predicted entries can directly measure).
  - Wilson-score 95% CIs throughout, not normal-approximation intervals,
    since several classes have small n and/or n_correct at or near the
    boundary (0 or n), where normal-approximation CIs misbehave.
  - The other_protein_recall_audit leakage rate: the fraction of
    other_protein predictions in that stratum corrected to a more
    specific class, i.e. an estimate of how much true-specific-class
    content the catch-all category is silently absorbing.

Usage
-----
    python scripts/validate_antigen_classifier_score.py --config configs/config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, require_path, log


def wilson_ci(n_success, n_total, z=1.96):
    if n_total == 0:
        return (None, None)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--in_path", default=None,
                         help="Default: <work_dir>/tables/antigen_classifier_sample_for_review.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    in_path = args.in_path or os.path.join(work_dir, "tables", "antigen_classifier_sample_for_review.csv")
    require_path(in_path, "hand-labeled antigen_classifier_sample_for_review.csv")

    df = pd.read_csv(in_path, dtype=str).fillna("")
    required = {"stratum", "antigen_name", "assigned_class", "correct_class"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[SCHEMA MISMATCH] missing {missing}")

    df["ground_truth"] = df["correct_class"].where(df["correct_class"] != "", df["assigned_class"])
    df["is_correct"] = df["ground_truth"] == df["assigned_class"]
    n_blank_corrections = int((df["correct_class"] == "").sum())
    log(f"{n_blank_corrections}/{len(df)} rows have no correct_class entered "
        f"(treated as assigned_class was correct) -- sanity-check this isn't "
        f"just an unfinished review before trusting the numbers below.")

    random_df = df[df["stratum"] == "random"]
    rare_df = df[df["stratum"] == "rare_class_oversample"]
    audit_df = df[df["stratum"] == "other_protein_recall_audit"]
    log(f"Strata sizes: random={len(random_df)}, rare_class_oversample={len(rare_df)}, "
        f"other_protein_recall_audit={len(audit_df)}")

    # --- Overall accuracy: random stratum ONLY ---
    n_correct = int(random_df["is_correct"].sum())
    n_total = len(random_df)
    acc = n_correct / n_total if n_total else None
    acc_ci = wilson_ci(n_correct, n_total)
    acc_str = f"{acc:.1%}" if acc is not None else "N/A"
    acc_ci_str = f"[{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]" if acc_ci[0] is not None else "N/A"
    log(f"Overall accuracy (random stratum, prevalence-weighted): "
        f"{n_correct}/{n_total} = {acc_str}, 95% CI {acc_ci_str}")

    # --- Confusion matrix: all strata combined ---
    all_classes = sorted(set(df["assigned_class"]) | set(df["ground_truth"]))
    confusion = pd.crosstab(df["assigned_class"], df["ground_truth"], dropna=False)
    confusion = confusion.reindex(index=all_classes, columns=all_classes, fill_value=0)
    log("Confusion matrix (rows=predicted, cols=ground truth):")
    log(confusion.to_string())

    # --- Per-class precision/recall/F1: random + rare_class_oversample combined ---
    combined = pd.concat([random_df, rare_df])
    per_class = {}
    for cls in all_classes:
        pred_cls = combined[combined["assigned_class"] == cls]
        n_pred = len(pred_cls)
        n_pred_correct = int(pred_cls["is_correct"].sum())
        precision = n_pred_correct / n_pred if n_pred else None
        precision_ci = wilson_ci(n_pred_correct, n_pred) if n_pred else (None, None)

        true_cls = combined[combined["ground_truth"] == cls]
        n_true = len(true_cls)
        n_true_correct = int((true_cls["assigned_class"] == cls).sum())
        recall = n_true_correct / n_true if n_true else None
        recall_ci = wilson_ci(n_true_correct, n_true) if n_true else (None, None)

        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None

        per_class[cls] = {
            "n_predicted": n_pred, "precision": precision, "precision_ci": list(precision_ci),
            "n_true_in_sample": n_true, "recall": recall, "recall_ci": list(recall_ci),
            "f1": f1,
        }
        log(f"  {cls}: precision={precision}, CI={precision_ci}, n_pred={n_pred} | "
            f"recall={recall}, CI={recall_ci}, n_true_in_sample={n_true}")

    # --- other_protein recall audit: leakage rate ---
    n_audit = len(audit_df)
    n_leaked = int((audit_df["ground_truth"] != "other_protein").sum())
    leakage_rate = n_leaked / n_audit if n_audit else None
    leakage_ci = wilson_ci(n_leaked, n_audit) if n_audit else (None, None)
    leaked_to = audit_df.loc[audit_df["ground_truth"] != "other_protein", "ground_truth"].value_counts().to_dict()
    leakage_str = f"{leakage_rate:.1%}" if leakage_rate is not None else "N/A"
    log(f"other_protein recall audit: {n_leaked}/{n_audit} ({leakage_str}) "
        f"corrected to a more specific class, 95% CI {leakage_ci}")
    log(f"  Leaked to: {leaked_to}")

    report = {
        "n_blank_corrections": n_blank_corrections,
        "overall_accuracy_random_stratum": {
            "n_correct": n_correct, "n_total": n_total, "accuracy": acc, "ci_95": list(acc_ci),
        },
        "confusion_matrix": confusion.to_dict(),
        "per_class_precision_recall_f1": per_class,
        "other_protein_recall_audit": {
            "n_audited": n_audit, "n_leaked_to_specific_class": n_leaked,
            "leakage_rate": leakage_rate, "leakage_rate_ci_95": list(leakage_ci),
            "leaked_to_breakdown": leaked_to,
        },
    }
    out_path = os.path.join(work_dir, "tables", "antigen_classifier_validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
