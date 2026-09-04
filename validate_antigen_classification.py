"""
validate_antigen_classification.py

Two-step antigen-classification validation workflow, run as two sequential
--step calls (a manual hand-labeling step is required between them and
cannot be automated away).

Superseded by validate_antigen_classifier_sample.py +
validate_antigen_classifier_score.py, which fix a sampling-design issue
here (this script samples grouped by the predicted class, which does not
support a population-weighted accuracy estimate -- see the newer scripts'
docstrings for the full rationale). Kept for reference.

  --step sample
      Draws a stratified random sample across antigen classes from the
      antigen_class column already persisted on master_antibodies.csv (see
      migrate_master_csv.py --only antigen_class if that column doesn't
      exist yet). Writes a CSV for manual review. Does not re-implement
      classification logic -- it reads the same antigen_class assignment
      the main pipeline already computed, so what is being validated is
      exactly the heuristic actually used in the paper.

  [ -- hand-edit the CSV here, filling in correct_class for any wrong
       rows, before running --step score -- ]

  --step score
      Reads the hand-reviewed CSV and computes overall + per-class
      precision for the antigen_class keyword heuristic.

Usage:
    python validate_antigen_classification.py --config config.yaml --step sample --n_per_class 15
    # ... hand-edit tables/antigen_classification_sample_FOR_REVIEW.csv ...
    python validate_antigen_classification.py --config config.yaml --step score
"""
import argparse
import json
import os

import pandas as pd
import yaml


def run_sample(master_path, out_dir, n_per_class, seed):
    out_path = os.path.join(out_dir, "antigen_classification_sample_FOR_REVIEW.csv")

    if not os.path.exists(master_path):
        raise SystemExit(f"[FILE NOT FOUND] {master_path} -- pass --master_csv if it lives elsewhere.")

    df = pd.read_csv(master_path)

    required_cols = {"antigen_name", "antigen_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[SCHEMA MISMATCH] master_antibodies.csv is missing {missing}.")

    if "antigen_class" not in df.columns:
        raise SystemExit(
            "[MISSING INPUT] No antigen_class column found. Run "
            "migrate_master_csv.py --only antigen_class first."
        )

    has_antigen = df[df["has_antigen"] == True].copy() if "has_antigen" in df.columns else df.copy()

    samples = []
    rng_state = seed
    for cls, group in has_antigen.groupby("antigen_class"):
        n = min(n_per_class, len(group))
        samples.append(group.sample(n=n, random_state=rng_state))
        rng_state += 1

    sample_df = pd.concat(samples, ignore_index=True)
    sample_df = sample_df[["antigen_name", "antigen_type", "antigen_class"]].rename(
        columns={"antigen_class": "assigned_class"}
    )
    sample_df["correct_class"] = ""
    sample_df["notes"] = ""

    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    sample_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(sample_df)} rows across {sample_df['assigned_class'].nunique()} classes to {out_path}")
    print("Next: open this CSV, fill in `correct_class` for any row where `assigned_class` is wrong "
          "(leave blank if assigned_class is correct), then run --step score")


def run_score(in_path, out_dir):
    out_path = os.path.join(out_dir, "antigen_classification_validation.json")

    if not os.path.exists(in_path):
        raise SystemExit(f"[FILE NOT FOUND] {in_path} -- run --step sample first.")

    df = pd.read_csv(in_path, dtype=str).fillna("")

    if "correct_class" not in df.columns:
        raise SystemExit("[SCHEMA MISMATCH] Expected a correct_class column. Run --step sample first.")

    df["ground_truth"] = df["correct_class"].where(df["correct_class"] != "", df["assigned_class"])
    df["is_correct"] = df["ground_truth"] == df["assigned_class"]

    n_total = len(df)
    n_correct = int(df["is_correct"].sum())
    overall_precision = n_correct / n_total if n_total else float("nan")

    per_class = (
        df.groupby("assigned_class")["is_correct"]
        .agg(n="count", n_correct="sum")
        .assign(precision=lambda d: d["n_correct"] / d["n"])
        .reset_index()
    )

    errors = df[~df["is_correct"]][["antigen_name", "antigen_type", "assigned_class", "ground_truth", "notes"]]

    result = {
        "n_reviewed": n_total,
        "n_correct": n_correct,
        "overall_precision": overall_precision,
        "per_class_precision": per_class.to_dict(orient="records"),
        "n_errors": len(errors),
        "errors": errors.to_dict(orient="records"),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[OK] Wrote {out_path}")
    print(f"\nOverall precision on {n_total} hand-reviewed rows: {overall_precision:.1%}")
    print("\nPer-class precision:")
    print(per_class.to_string(index=False))
    if len(errors):
        print(f"\n{len(errors)} misclassification(s) found -- see 'errors' in the output JSON.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", required=True, choices=["sample", "score"])
    ap.add_argument("--master_csv", default=None,
                     help="--step sample: override path. Default: <work_dir>/tables/master_antibodies.csv")
    ap.add_argument("--in_path", default=None,
                     help="--step score: override input CSV path. Default: "
                          "<work_dir>/tables/antigen_classification_sample_FOR_REVIEW.csv")
    ap.add_argument("--out_dir", default=None, help="Override output dir. Default: <work_dir>/tables")
    ap.add_argument("--n_per_class", type=int, default=15,
                     help="--step sample: max rows to sample per antigen_class category")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    work_dir = config["paths"]["work_dir"]
    out_dir = args.out_dir or os.path.join(work_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    if args.step == "sample":
        master_path = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")
        run_sample(master_path, out_dir, args.n_per_class, args.seed)
    elif args.step == "score":
        in_path = args.in_path or os.path.join(out_dir, "antigen_classification_sample_FOR_REVIEW.csv")
        run_score(in_path, out_dir)


if __name__ == "__main__":
    main()
