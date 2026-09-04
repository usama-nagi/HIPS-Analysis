#!/usr/bin/env python3
"""
audit_dedup_sensitivity.py

This script provides two robustness checks, reusing
compute_diversity_snapshot() imported directly from the existing module
(not reimplemented, so results cannot drift from RQ3's own logic):

  CHECK 1 -- Outcome-independent representative selection + sensitivity.
  Two alternative selection rules, neither conditioned on has_antigen:
    (a) deterministic: smallest pdb_id only (no has_antigen tie-break)
    (b) random: uniform random draw within each cluster, repeated over
        N_SEEDS different seeds, reporting the spread (min/mean/max/std)
        of every headline metric across draws -- i.e. how much each
        metric moves if the representative is picked differently.
  All three methods (original has_antigen-preferring, new deterministic,
  new random-sweep) are reported side by side for direct comparison.

  CHECK 2 -- Non-circular antigen diversity measure. Rather than asking
  "how many post-dedup representatives are antigen-bound," count unique
  (CDR-H3 cluster, antigen cluster) combinations directly among
  antigen-bound rows -- this measures antigen diversity conditional on
  CDR-H3 redundancy without any representative-selection step (and
  therefore without any possible circularity from how a representative
  was chosen) at all.

Usage
-----
    python scripts/audit_dedup_sensitivity.py --config configs/config.yaml \
        --n_seeds 200
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

# Reuse the REAL diversity-metric logic and the REAL original selection
# rule directly from the existing module, rather than reimplementing them
# (avoids any risk of silently drifting from RQ3's own numbers).
from rq3_redundancy_and_recommendations import (
    pick_cluster_representatives, compute_diversity_snapshot,
)


def pick_representatives_deterministic_no_antigen_bias(merged: pd.DataFrame) -> pd.DataFrame:
    """Same shape as the original pick_cluster_representatives(), but the
    tie-break is smallest pdb_id ONLY -- has_antigen plays no role at all,
    so antigen retention among the result is not selection-influenced."""
    sorted_df = merged.sort_values(by=["cluster_rep", "pdb_id"], ascending=[True, True])
    return sorted_df.drop_duplicates(subset=["cluster_rep"], keep="first")


def pick_representatives_random(merged: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Uniform random representative per cluster -- has_antigen plays no
    role; used to sweep over many seeds and report the resulting spread."""
    rng = np.random.default_rng(seed)
    # shuffle then keep first per group == uniform random pick per group
    shuffled = merged.sample(frac=1.0, random_state=seed)
    return shuffled.drop_duplicates(subset=["cluster_rep"], keep="first")


def h3_x_antigen_cluster_combinations(master_df: pd.DataFrame) -> dict:
    """count unique (CDR-H3 cluster, antigen cluster) combinations
    among antigen-bound rows directly -- no representative selection step,
    so no possible circularity. This is the "diversity conditional on H3
    redundancy" measure that replaces the old, circular
    has_antigen-preferring-representative approach."""
    ag = master_df[master_df["has_antigen"] == True].dropna(
        subset=["cluster_rep", "antigen_cluster_id"]
    )
    combos = ag[["cluster_rep", "antigen_cluster_id"]].drop_duplicates()
    return {
        "n_antigen_bound_rows": int(len(ag)),
        "n_raw_antigen_clusters_unconditional": int(ag["antigen_cluster_id"].nunique()),
        "n_h3xantigen_combinations": int(len(combos)),
        "n_h3_clusters_represented": int(combos["cluster_rep"].nunique()),
        "note": "n_h3xantigen_combinations is the non-circular replacement for "
                "'antigen clusters retained after dedup' -- it measures antigen "
                "diversity conditional on CDR-H3 clustering directly, without "
                "picking any single representative per cluster.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--n_seeds", type=int, default=200,
                         help="Number of random-draw repeats for the sensitivity sweep.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    master_csv = os.path.join(work_dir, "tables", "master_antibodies.csv")
    clusters_tsv = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    require_path(master_csv, "master_antibodies.csv")
    require_path(clusters_tsv, "rq1_cdrh3_clusters.tsv")

    master_df = pd.read_csv(master_csv, low_memory=False)
    clusters_df = pd.read_csv(clusters_tsv, sep="\t")
    if "cluster_rep" in master_df.columns:
        master_df = master_df.drop(columns=["cluster_rep"])
    merged = master_df.merge(
        clusters_df[["filename_stem", "cluster_rep"]], on="filename_stem", how="inner"
    )
    log(f"Loaded {len(master_df)} rows, {clusters_df['cluster_rep'].nunique()} CDR-H3 clusters, "
        f"{len(merged)} rows with a cluster assignment")

    # --- Check 2: non-circular antigen diversity measure (no representative selection at all) ---
    log("=" * 70)
    log("Check 2: H3-cluster x antigen-cluster combinations (non-circular)")
    combo_result = h3_x_antigen_cluster_combinations(merged)
    log(json.dumps(combo_result, indent=2))

    # --- Check 1, baseline: reproduce the has_antigen-preferring method as a sanity check ---
    log("=" * 70)
    log("Reproducing has_antigen-preferring selection (sanity check vs. paper)")
    original_dedup = pick_cluster_representatives(master_df, clusters_df)
    original_snapshot = compute_diversity_snapshot(original_dedup)
    log(f"Baseline method: n={len(original_dedup)}, "
        f"antigen_cluster_gini={original_snapshot.get('antigen_cluster_gini')}, "
        f"antigen_cluster_n_unique={original_snapshot.get('antigen_cluster_n_unique')} "
        f"(paper reports n=5,365, antigen Gini=0.535, n_unique=1,741)")

    # --- Check 1a: deterministic, outcome-independent (no has_antigen bias) ---
    log("=" * 70)
    log("Check 1a: deterministic, outcome-independent selection (pdb_id only, no has_antigen bias)")
    det_dedup = pick_representatives_deterministic_no_antigen_bias(merged)
    det_snapshot = compute_diversity_snapshot(det_dedup)
    log(f"Deterministic no-bias method: n={len(det_dedup)}, "
        f"antigen_cluster_gini={det_snapshot.get('antigen_cluster_gini')}, "
        f"antigen_cluster_n_unique={det_snapshot.get('antigen_cluster_n_unique')}")

    # --- Check 1b: random sweep over N_SEEDS, report spread ---
    log("=" * 70)
    log(f"Check 1b: random-draw sensitivity sweep, {args.n_seeds} seeds")
    sweep_metrics = {
        "antigen_cluster_gini": [], "antigen_cluster_n_unique": [],
        "length_gini": [], "heavy_germline_gini": [], "therapeutic_fraction": [],
        "n_antibodies": [],
    }
    for seed in range(args.n_seeds):
        draw_df = pick_representatives_random(merged, seed)
        snap = compute_diversity_snapshot(draw_df)
        for k in sweep_metrics:
            if k in snap and snap[k] is not None:
                sweep_metrics[k].append(snap[k])
        if (seed + 1) % 50 == 0:
            log(f"  {seed + 1}/{args.n_seeds} draws done")

    sweep_summary = {}
    for k, vals in sweep_metrics.items():
        if vals:
            arr = np.array(vals)
            sweep_summary[k] = {
                "n_draws": len(arr), "min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean()), "std": float(arr.std()),
            }
    log(json.dumps(sweep_summary, indent=2))

    report = {
        "non_circular_antigen_diversity": combo_result,
        "original_circular_method": {
            "n_representatives": len(original_dedup),
            "antigen_cluster_gini": original_snapshot.get("antigen_cluster_gini"),
            "antigen_cluster_n_unique": original_snapshot.get("antigen_cluster_n_unique"),
            "length_gini": original_snapshot.get("length_gini"),
        },
        "deterministic_no_antigen_bias_method": {
            "n_representatives": len(det_dedup),
            "antigen_cluster_gini": det_snapshot.get("antigen_cluster_gini"),
            "antigen_cluster_n_unique": det_snapshot.get("antigen_cluster_n_unique"),
            "length_gini": det_snapshot.get("length_gini"),
        },
        "random_sweep_sensitivity": sweep_summary,
    }
    out_path = os.path.join(work_dir, "tables", "dedup_circularity_check.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
