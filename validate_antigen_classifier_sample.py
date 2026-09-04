#!/usr/bin/env python3
"""
validate_antigen_classifier_sample.py
============================
Produces a CSV for manual hand-labeling; the actual labeling step cannot
be automated (the same limitation validate_antigen_classification.py
documents). This script's sampling design addresses three limitations of
a simpler predicted-class-grouped sample:

  (1) Drawing n_per_class rows grouped by the predicted class
      (antigen_class) produces a sample balanced across classes regardless
      of true prevalence. Any accuracy computed from that sample is
      macro-averaged across classes, not population-weighted -- distinct
      from, and not a substitute for, the true corpus-wide accuracy given
      the real ~64%/27%/<3%-per-class prevalence.

  (2) Correcting mistakes found in a sample and then treating that same
      (now partially hand-corrected) sample as validating "the classifier"
      validates a hybrid of the heuristic and the manual correction step,
      not the heuristic alone, so a genuinely independent validation set
      is needed.

  (3) Precision-only evaluation says nothing about recall: an entry whose
      true class is, say, "enzyme" but was assigned to the "other_protein"
      catch-all would never surface as an error in a sample grouped by
      predicted class, since it would only be flagged if the reviewer
      already suspects the true class might be something else. A
      dedicated audit for the catch-all class addresses this directly.

This script draws three independent, non-overlapping strata, output as one
CSV with a `stratum` column so the scoring script can combine them
correctly:

  stratum="random"                  -- a genuinely random sample across all
      antigen-bound entries, unconditioned on the predicted class. This is
      the only stratum from which population-weighted (prevalence-correct)
      accuracy can be validly computed, since it's an unbiased sample of
      the true class-prevalence distribution.
  stratum="rare_class_oversample"   -- an additional stratified sample
      over-representing the small predicted classes (anything other than
      the two dominant ones), so per-class precision for rare classes has
      a usable sample size instead of the ~3-9 rare-class rows a pure
      random draw of a few hundred would produce. Not used for
      population-weighted accuracy (would double-count and bias it) --
      used only for per-class precision/recall within this stratum.
  stratum="other_protein_recall_audit" -- a sample of entries predicted
      other_protein specifically, checking whether any should have
      matched a more specific class the keyword heuristic missed. This
      measures how much true-specific-class content is absorbed into the
      catch-all bucket, which no precision-only, grouped-by-predicted-class
      sample can surface.

All three strata exclude the filename_stems already reviewed in the
existing antigen_classification_sample_FOR_REVIEW.csv (if present), so
this is a genuinely independent set, not a re-review of already-corrected
rows.

Usage
-----
    python scripts/validate_antigen_classifier_sample.py --config configs/config.yaml \
        --n_random 500 --n_rare_class 30 --n_other_protein_audit 100
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, require_path, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--n_random", type=int, default=500,
                         help="Pure random sample size, unconditioned on predicted class.")
    parser.add_argument("--n_rare_class", type=int, default=30,
                         help="Per-class oversample size for classes other than the two dominant ones.")
    parser.add_argument("--n_other_protein_audit", type=int, default=100,
                         help="Sample size of other_protein-predicted entries for the recall audit.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")

    df = pd.read_csv(master_path, low_memory=False)
    required = {"antigen_name", "antigen_type", "has_antigen"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[SCHEMA MISMATCH] missing {missing}. Found: {list(df.columns)}")
    if "antigen_class" not in df.columns:
        raise SystemExit(
            "[MISSING INPUT] No antigen_class column. Run "
            "migrate_master_csv.py --only antigen_class first."
        )

    ag_df = df[df["has_antigen"] == True].copy()
    log(f"Antigen-bound entries: {len(ag_df)}")

    # Real prevalence, computed fresh rather than assumed
    prevalence = ag_df["antigen_class"].value_counts(normalize=True).sort_values(ascending=False)
    log("Real predicted-class prevalence:")
    for cls, frac in prevalence.items():
        log(f"  {cls}: {100*frac:.2f}%")
    dominant_classes = set(prevalence.head(2).index)
    log(f"Dominant classes (excluded from rare-class oversample): {dominant_classes}")

    # Real count for the nucleic-acid/carbohydrate MMseqs2-settings question
    n_nucleic_carb = int((ag_df["antigen_class"] == "nucleic_acid_or_carbohydrate").sum()) \
        if "nucleic_acid_or_carbohydrate" in ag_df["antigen_class"].values else 0
    log(f"nucleic_acid_or_carbohydrate entries (protein-tuned MMseqs2 settings currently "
        f"applied uniformly, per assign_antigen_clusters.py -- no molecule-type branch found): "
        f"{n_nucleic_carb}")

    # Exclude anything already reviewed in the existing sample, for genuine independence
    prior_review_path = os.path.join(work_dir, "tables", "antigen_classification_sample_FOR_REVIEW.csv")
    excluded_names = set()
    if os.path.exists(prior_review_path):
        prior = pd.read_csv(prior_review_path)
        excluded_names = set(prior["antigen_name"].astype(str))
        log(f"Excluding {len(excluded_names)} previously-reviewed antigen_name values "
            f"from {prior_review_path} for independence")
    pool = ag_df[~ag_df["antigen_name"].astype(str).isin(excluded_names)]
    log(f"Pool available after exclusion: {len(pool)}")

    rng_state = args.seed
    all_rows = []

    # --- Stratum 1: pure random, unconditioned on predicted class ---
    n = min(args.n_random, len(pool))
    random_sample = pool.sample(n=n, random_state=rng_state)
    random_sample = random_sample.assign(stratum="random")
    all_rows.append(random_sample)
    log(f"Stratum 'random': {len(random_sample)} rows "
        f"(class breakdown: {random_sample['antigen_class'].value_counts().to_dict()})")
    rng_state += 1

    # --- Stratum 2: rare-predicted-class oversample ---
    remaining_pool = pool[~pool.index.isin(random_sample.index)]
    rare_rows = []
    for cls, group in remaining_pool.groupby("antigen_class"):
        if cls in dominant_classes:
            continue
        n_cls = min(args.n_rare_class, len(group))
        rare_rows.append(group.sample(n=n_cls, random_state=rng_state))
        rng_state += 1
    if rare_rows:
        rare_sample = pd.concat(rare_rows).assign(stratum="rare_class_oversample")
        all_rows.append(rare_sample)
        log(f"Stratum 'rare_class_oversample': {len(rare_sample)} rows across "
            f"{rare_sample['antigen_class'].nunique()} classes")

    # --- Stratum 3: other_protein recall audit ---
    remaining_pool2 = remaining_pool[~remaining_pool.index.isin(
        pd.concat(rare_rows).index if rare_rows else pd.Index([])
    )]
    other_protein_pool = remaining_pool2[remaining_pool2["antigen_class"] == "other_protein"]
    n_audit = min(args.n_other_protein_audit, len(other_protein_pool))
    audit_sample = other_protein_pool.sample(n=n_audit, random_state=rng_state).assign(
        stratum="other_protein_recall_audit"
    )
    all_rows.append(audit_sample)
    log(f"Stratum 'other_protein_recall_audit': {len(audit_sample)} rows")

    combined = pd.concat(all_rows, ignore_index=True)
    out_df = combined[["stratum", "antigen_name", "antigen_type", "antigen_class"]].rename(
        columns={"antigen_class": "assigned_class"}
    )
    out_df["correct_class"] = ""
    out_df["notes"] = ""
    out_df = out_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_path = os.path.join(work_dir, "tables", "antigen_classifier_sample_for_review.csv")
    out_df.to_csv(out_path, index=False)
    log(f"Wrote {len(out_df)} rows to {out_path}")

    meta_path = os.path.join(work_dir, "tables", "antigen_classifier_sample_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "real_prevalence": prevalence.to_dict(),
            "dominant_classes": list(dominant_classes),
            "n_nucleic_acid_or_carbohydrate": n_nucleic_carb,
            "n_excluded_prior_review": len(excluded_names),
        }, f, indent=2)
    log(f"Wrote {meta_path}")

    print("\n" + "=" * 70)
    print("NEXT STEP (manual, cannot be automated):")
    print(f"Open {out_path} and fill in `correct_class` for every row where")
    print("`assigned_class` is wrong. Leave blank if assigned_class is correct.")
    print("For stratum='other_protein_recall_audit' rows specifically: fill in")
    print("correct_class if the antigen_name suggests a MORE SPECIFIC class")
    print("(viral, enzyme, immune_receptor, bacterial, cancer-associated,")
    print("nucleic_acid_or_carbohydrate) that the heuristic missed, even if")
    print("'other_protein' is a defensible generic label.")
    print("Then run validate_antigen_classifier_score.py.")


if __name__ == "__main__":
    main()
