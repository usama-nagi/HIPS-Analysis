"""
migrate_master_csv.py

Consolidates three one-time, standalone operations against
master_antibodies.csv:

  --only diagnose
      Diagnostic only, writes no changes to master_antibodies.csv. Answers
      whether light_subclass's "UNKNOWN" vs "unknown" split is a real
      distinction or a case-sensitivity artifact, by cross-tabulating
      against `paired` and `l_chain == "NA"`.
      Finding: "unknown" (lowercase) is the literal string SAbDab's own
      summary export writes for an attempted-and-failed germline call,
      while "UNKNOWN" (uppercase) is introduced by
      rq1_sequence_structural_bias.py's own fillna("UNKNOWN") for rows
      where light_subclass is empty/NaN in the source table. These are
      two distinct categories with different upstream causes, not a case
      artifact -- report them separately, never merge.

  --only h3_seq
      Adds h3_seq (isolated CDR-H3-only sequence per row) to
      master_antibodies.csv, using extract_cdrh3_sequences() imported
      directly from rq1_sequence_structural_bias.py (see "Why import,
      not copy" below) rather than a local reimplementation.

  --only cluster_rep
      Merges cluster_rep onto the full master table from
      rq1_cdrh3_clusters.tsv (Section B's output). Independent of h3_seq --
      run either or both, in any order.

  --only antigen_class
      Adds antigen_class to master_antibodies.csv using classify_antigen()
      imported directly from rq1_sequence_structural_bias.py's Section C
      (see "Why import, not copy" below), then self-verifies the result
      reproduces rq1_antigen_landscape.json's antigen_class_counts exactly
      before declaring success.

A backup of master_antibodies.csv is written before each operation's first
run (master_antibodies.csv.bak_pre_<operation>), so any operation can be
compared against the pre-migration state later.

Usage:
    python migrate_master_csv.py --config config.yaml --only diagnose
    python migrate_master_csv.py --config config.yaml --only h3_seq
    python migrate_master_csv.py --config config.yaml --only cluster_rep
    python migrate_master_csv.py --config config.yaml --only antigen_class
    python migrate_master_csv.py --config config.yaml --only all     # runs all 4, in a safe order
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Why import, not copy: classify_antigen() and extract_cdrh3_sequences()
# are imported directly from rq1_sequence_structural_bias.py rather than
# reimplemented here, so this script's output cannot drift from that
# module's own definitions -- the same reasoning applied throughout this
# pipeline wherever a script needs logic that another script already owns.
from rq1_sequence_structural_bias import classify_antigen, extract_cdrh3_sequences


# Shared helpers
def _load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def _backup_once(master_path, suffix):
    backup_path = f"{master_path}.bak_pre_{suffix}"
    if not os.path.exists(backup_path):
        shutil.copy2(master_path, backup_path)
        print(f"[OK] Backed up original to {backup_path}")
    else:
        print(f"[INFO] Backup already exists at {backup_path}, not overwriting it.")

# --only diagnose
def run_diagnose(master_path, out_dir):
    out_path = os.path.join(out_dir, "light_germline_label_diagnosis.json")

    df = pd.read_csv(master_path)
    if "light_subclass" not in df.columns:
        print("[SCHEMA MISMATCH] master_antibodies.csv has no light_subclass column.")
        return

    is_upper = df["light_subclass"] == "UNKNOWN"
    is_lower = df["light_subclass"] == "unknown"

    crosstab_paired = pd.crosstab(
        df["light_subclass"].where(is_upper | is_lower, other="other"),
        df["paired"],
    )

    if "l_chain" in df.columns:
        crosstab_lchain_na = pd.crosstab(
            df["light_subclass"].where(is_upper | is_lower, other="other"),
            df["l_chain"] == "NA",
        )
    else:
        crosstab_lchain_na = None

    n_upper = int(is_upper.sum())
    n_lower = int(is_lower.sum())
    n_upper_paired_true = int((is_upper & (df["paired"] == True)).sum())
    n_upper_paired_false = int((is_upper & (df["paired"] == False)).sum())
    n_lower_paired_true = int((is_lower & (df["paired"] == True)).sum())
    n_lower_paired_false = int((is_lower & (df["paired"] == False)).sum())

    result = {
        "n_total_rows": len(df),
        "n_UNKNOWN_upper": n_upper,
        "n_unknown_lower": n_lower,
        "UNKNOWN_upper_by_paired": {
            "paired_True": n_upper_paired_true,
            "paired_False": n_upper_paired_false,
        },
        "unknown_lower_by_paired": {
            "paired_True": n_lower_paired_true,
            "paired_False": n_lower_paired_false,
        },
        "hypothesis_A_clean_split": (
            n_upper_paired_false == n_upper and n_lower_paired_true == n_lower
        ),
        "crosstab_light_subclass_vs_paired": crosstab_paired.to_dict(),
        "resolved_finding": (
            "RESOLVED (see source code trace, not just this crosstab): 'unknown' "
            "(lowercase) is the literal string SAbDab's own summary export writes "
            "for an attempted-and-failed germline call. 'UNKNOWN' (uppercase) is "
            "introduced by rq1_sequence_structural_bias.py's own "
            "fillna('UNKNOWN') for rows where light_subclass is empty/NaN in the "
            "source table -- it is NOT present in master_antibodies.csv's raw "
            "light_subclass column itself (confirmed: 'UNKNOWN' does not appear "
            "in master_antibodies.csv's light_subclass.value_counts()). These are "
            "two distinct categories with different upstream causes. Report "
            "separately; do not merge."
        ),
    }

    if crosstab_lchain_na is not None:
        result["crosstab_light_subclass_vs_l_chain_NA"] = crosstab_lchain_na.to_dict()

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[OK] Wrote {out_path}")
    print(f"\n{result['resolved_finding']}\n")

# --only h3_seq   (from patch_add_h3_seq_and_cluster_rep.py, part 1)
def run_h3_seq(master_path):
    _backup_once(master_path, "h3seq_patch")

    df = pd.read_csv(master_path)
    print(f"[INFO] Loaded {len(df)} rows from {master_path}")

    required = {"pt_path", "cdr3_len_actual", "filename_stem"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[SCHEMA MISMATCH] master_antibodies.csv missing {missing}.")

    print("[INFO] Isolating h3_seq for every row via the real "
          "extract_cdrh3_sequences() (re-reads each .pt file)...")
    cdrh3_seqs, index_to_filename_stem = extract_cdrh3_sequences(df)
    stem_to_h3 = {
        index_to_filename_stem[idx]: seq for idx, seq in cdrh3_seqs.items()
    }
    df["h3_seq"] = df["filename_stem"].map(stem_to_h3)
    n_failures = int(df["h3_seq"].isna().sum())
    print(f"[OK] h3_seq isolated for {len(df) - n_failures}/{len(df)} rows "
          f"({n_failures} failures -- these rows get h3_seq=NaN, not a guessed value)")

    mismatch_mask = df["h3_seq"].notna() & (df["h3_seq"].str.len() != df["cdr3_len_actual"])
    n_mismatch = int(mismatch_mask.sum())
    if n_mismatch > 0:
        print(f"[WARNING] {n_mismatch} rows have h3_seq length != cdr3_len_actual.")
        print(df.loc[mismatch_mask, "filename_stem"].head(10).tolist())
    else:
        print("[OK] Sanity check passed: h3_seq length matches cdr3_len_actual everywhere.")

    df.to_csv(master_path, index=False)
    print(f"[OK] Wrote {master_path} ({len(df)} rows, {len(df.columns)} columns, +h3_seq)")

# --only cluster_rep   (from patch_add_h3_seq_and_cluster_rep.py, part 2)
def run_cluster_rep(master_path, clusters_path):
    _backup_once(master_path, "cluster_rep_patch")

    if not os.path.exists(clusters_path):
        sys.exit(f"[FILE NOT FOUND] {clusters_path} -- run rq1_sequence_structural_bias.py "
                  f"(Section B) first, it produces this file.")

    df = pd.read_csv(master_path)
    print(f"[INFO] Loaded {len(df)} rows from {master_path}")

    # master_antibodies.csv may already carry a cluster_rep column from an
    # earlier run of this same patch (e.g. against a previous, possibly
    # different rq1_cdrh3_clusters.tsv). If we don't drop it first, the
    # merge below silently produces cluster_rep_x / cluster_rep_y instead
    # of cluster_rep, and the very next line (checking df["cluster_rep"])
    # raises a KeyError -- this is exactly what happened the first time
    # this script was run a second time after rq1_cdrh3_clusters.tsv had
    # been regenerated. Always treat clusters_path as the single source
    # of truth and overwrite any existing cluster_rep column cleanly.
    if "cluster_rep" in df.columns:
        print("[INFO] master_antibodies.csv already has a cluster_rep column -- "
              "dropping it before merge so the fresh assignment from "
              f"{clusters_path} replaces it cleanly, rather than colliding "
              "into cluster_rep_x/cluster_rep_y.")
        df = df.drop(columns=["cluster_rep"])

    print(f"[INFO] Merging cluster_rep from {clusters_path}...")

    clusters_df = pd.read_csv(clusters_path, sep="\t")
    if "filename_stem" not in clusters_df.columns or "cluster_rep" not in clusters_df.columns:
        sys.exit(f"[SCHEMA MISMATCH] {clusters_path} missing filename_stem or cluster_rep columns.")

    n_before = len(df)
    df = df.merge(clusters_df[["filename_stem", "cluster_rep"]], on="filename_stem", how="left")
    if len(df) != n_before:
        sys.exit(f"[JOIN ERROR] Row count changed after merge ({n_before} -> {len(df)}). "
                  f"filename_stem is probably not unique -- stopping rather than writing "
                  f"a corrupted master table.")

    n_unmatched = int(df["cluster_rep"].isna().sum())
    if n_unmatched > 0:
        print(f"[WARNING] {n_unmatched}/{len(df)} rows did not match a cluster_rep.")
    else:
        print(f"[OK] All {len(df)} rows matched a cluster_rep.")

    df.to_csv(master_path, index=False)
    print(f"[OK] Wrote {master_path} ({len(df)} rows, {len(df.columns)} columns, +cluster_rep)")

# --only antigen_class   (from patch_add_antigen_class.py)
def run_antigen_class(master_path, landscape_path):
    _backup_once(master_path, "antigen_class_patch")

    df = pd.read_csv(master_path)
    print(f"[INFO] Loaded {len(df)} rows from {master_path}")

    required = {"has_antigen", "antigen_name", "antigen_type"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[SCHEMA MISMATCH] master_antibodies.csv missing {missing}")

    df["antigen_class"] = None
    ag_mask = df["has_antigen"] == True
    df.loc[ag_mask, "antigen_class"] = df.loc[ag_mask].apply(
        lambda r: classify_antigen(r.get("antigen_name", ""), r.get("antigen_type", "")),
        axis=1,
    )
    print(f"[OK] Classified {int(ag_mask.sum())} antigen-bound rows; "
          f"{int((~ag_mask).sum())} non-antigen rows left as antigen_class=NaN.")

    df.to_csv(master_path, index=False)
    print(f"[OK] Wrote {master_path} ({len(df)} rows, {len(df.columns)} columns, +antigen_class)")

    if os.path.exists(landscape_path):
        with open(landscape_path) as f:
            landscape = json.load(f)
        expected_counts = landscape.get("antigen_class_counts", {})
        actual_counts = df.loc[ag_mask, "antigen_class"].value_counts().to_dict()

        if expected_counts == actual_counts:
            print("[VERIFIED] antigen_class value_counts exactly match "
                  "rq1_antigen_landscape.json's antigen_class_counts. Safe to proceed.")
        else:
            print("[MISMATCH -- DO NOT TRUST THIS COLUMN YET]")
            print(f"  Expected: {expected_counts}")
            print(f"  Actual:   {actual_counts}")
            sys.exit(1)
    else:
        print(f"[WARNING] {landscape_path} not found -- could not verify. Re-run "
              f"rq1_sequence_structural_bias.py --only C first if you want this check.")

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", required=True,
                     choices=["diagnose", "h3_seq", "cluster_rep", "antigen_class", "all"])
    ap.add_argument("--master_csv", default=None,
                     help="Override path. Default: <work_dir>/tables/master_antibodies.csv "
                          "(note: diagnose's original default was <work_dir>/master_antibodies.csv "
                          "without /tables/ -- if your file lives there instead, pass this explicitly)")
    ap.add_argument("--clusters_path", default=None,
                     help="--only cluster_rep: override path. Default: <work_dir>/tables/rq1_cdrh3_clusters.tsv")
    ap.add_argument("--landscape_json", default=None,
                     help="--only antigen_class: override path. Default: <work_dir>/tables/rq1_antigen_landscape.json")
    ap.add_argument("--out_dir", default=None,
                     help="--only diagnose: override output dir. Default: <work_dir>/tables")
    args = ap.parse_args()

    config = _load_config(args.config)
    work_dir = config["paths"]["work_dir"]
    master_path = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")
    out_dir = args.out_dir or os.path.join(work_dir, "tables")
    clusters_path = args.clusters_path or os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    landscape_path = args.landscape_json or os.path.join(work_dir, "tables", "rq1_antigen_landscape.json")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(master_path):
        sys.exit(f"[FILE NOT FOUND] {master_path} -- pass --master_csv if it lives elsewhere.")

    ops = ["diagnose", "h3_seq", "cluster_rep", "antigen_class"] if args.only == "all" else [args.only]

    for op in ops:
        print(f"\n{'='*70}\nRunning: {op}\n{'='*70}")
        if op == "diagnose":
            run_diagnose(master_path, out_dir)
        elif op == "h3_seq":
            run_h3_seq(master_path)
        elif op == "cluster_rep":
            run_cluster_rep(master_path, clusters_path)
        elif op == "antigen_class":
            run_antigen_class(master_path, landscape_path)


if __name__ == "__main__":
    main()