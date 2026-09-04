#!/usr/bin/env python3
"""
audit_antigen_bound_denominators.py

Reuses pick_cluster_representatives() imported directly from
rq3_redundancy_and_recommendations.py -- the SAME function that produced
the 5,404-representative set already reported in Table tab:dedup and
sanity-checked in audit_dedup_sensitivity.py -- so this number cannot
silently diverge from what's already published.

Usage
-----
    python scripts/audit_antigen_bound_denominators.py --config configs/config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

from rq3_redundancy_and_recommendations import pick_cluster_representatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    master_csv = os.path.join(work_dir, "tables", "master_antibodies.csv")
    clusters_tsv = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    require_path(master_csv, "master_antibodies.csv")
    require_path(clusters_tsv, "rq1_cdrh3_clusters.tsv")

    master_df = pd.read_csv(master_csv, low_memory=False)
    clusters_df = pd.read_csv(clusters_tsv, sep="\t")

    if "has_antigen" not in master_df.columns:
        raise ValueError(
            f"[SCHEMA MISMATCH] no has_antigen column. Found: {list(master_df.columns)}"
        )

    n_before = int((master_df["has_antigen"] == True).sum())
    log(f"Antigen-bound entries, full corpus: {n_before} "
        f"(cross-check against the widely-cited 15,736)")

    representatives = pick_cluster_representatives(master_df, clusters_df)
    n_after = int((representatives["has_antigen"] == True).sum())
    log(f"Post-dedup representatives: {len(representatives)} total "
        f"(cross-check against 5,404)")
    log(f"Antigen-bound entries among post-dedup representatives: {n_after}")

    n_unique_antigen_clusters_after = representatives.loc[
        representatives["has_antigen"] == True, "antigen_cluster_id"
    ].nunique()
    log(f"Sanity check: n_unique antigen clusters among these {n_after} antigen-bound "
        f"representatives = {n_unique_antigen_clusters_after} "
        f"(cross-check against 1,747 -- must be <= n_after, since multiple "
        f"representatives can share one antigen cluster)")

    report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_representatives_total": len(representatives),
        "n_unique_antigen_clusters_after_sanity_check": int(n_unique_antigen_clusters_after),
    }
    out_path = os.path.join(work_dir, "tables", "antigen_bound_denominator_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
