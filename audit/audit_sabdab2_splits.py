#!/usr/bin/env python3
"""
audit_sabdab2_splits.py
========================
SAbDab2 (Capel et al. 2026) publishes standardized
ab-split and ab-ag-split partitions that explicitly control antibody- and
antigen-similarity leakage. This script tests two distinct questions against
those real, published splits:

  (1) Leakage consistency -- using relations derived independently from
      this project's own pipeline (CDR-H3 cluster, antigen cluster, PDB
      identifier, exact VH/VL sequence identity), not SAbDab2's own cluster
      columns. Checking overlap against SAbDab2's own clusters would be
      circular (SAbDab2's split respects SAbDab2's own clusters by
      construction) and would prove nothing. This mirrors the
      CHIMERA-Bench audit's external-relation methodology exactly.

  (2) Representativeness drift -- does composition (antigen class, antigen-
      cluster concentration) differ between train and test even where
      leakage is genuinely controlled? This is the direct, external,
      real-data test of this paper's central thesis: leakage control and
      biological representativeness are separable properties.

master_antibodies.csv already carries cluster_rep and antigen_class as
precomputed columns, so this script uses those directly.

Reads:
  <splits_dir>/ab_split.csv, <splits_dir>/abag_split.csv   (from SAbDab2/Zenodo)
  <work_dir>/tables/master_antibodies.csv

Writes:
  <work_dir>/tables/audit_sabdab2_splits.json

Usage
-----
    python scripts/audit_sabdab2_splits.py --config configs/config.yaml \
        --splits_dir splits_final
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, shannon_entropy, gini_coefficient, require_path, log

REQUIRED_MASTER_COLS = [
    "pdb_id", "h_chain", "l_chain", "has_antigen", "antigen_cluster_id",
    "antigen_class", "cluster_rep", "heavy_seq", "light_seq",
]


def clean_pdb_id(raw: str) -> str:
    """'pdb_00001ejo' -> '1ejo'. SAbDab2 zero-pads the PDB_ID field to 8
    chars after the 'pdb_' prefix; classic 4-char codes get 4 leading
    zeros, which we strip. A genuine 8-char extended PDB ID (no leading
    zeros) is left as-is, since that's not a padding artifact."""
    if not isinstance(raw, str) or not raw.startswith("pdb_"):
        return raw
    code = raw[4:]
    if code[:4] == "0000":
        return code[4:]
    return code


def load_split(splits_dir: str, fname: str, split_col: str) -> pd.DataFrame:
    path = os.path.join(splits_dir, fname)
    require_path(path, "SAbDab2 split file (download from Zenodo record 20083995)")
    df = pd.read_csv(path, usecols=[
        "INSTANCE", "PDB_ID", "Hchain", "Lchain", "type", split_col
    ])
    df["pdb_std"] = df["PDB_ID"].apply(clean_pdb_id)
    return df


def join_to_our_population(sabdab2_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Join on (pdb, h_chain, l_chain), NaN-safe (single-domain entries have
    a NaN light chain on one or both sides)."""
    left = sabdab2_df.copy()
    right = master.copy()
    left["pdb_key"] = left["pdb_std"].fillna("__NONE__").astype(str)
    left["h_key"] = left["Hchain"].fillna("__NONE__").astype(str)
    left["l_key"] = left["Lchain"].fillna("__NONE__").astype(str)
    right["pdb_key"] = right["pdb_id"].fillna("__NONE__").astype(str)
    right["h_key"] = right["h_chain"].fillna("__NONE__").astype(str)
    right["l_key"] = right["l_chain"].fillna("__NONE__").astype(str)
    merged = left.merge(right, on=["pdb_key", "h_key", "l_key"], how="left", suffixes=("_s2", "_ours"))
    return merged


def leakage_check(merged: pd.DataFrame, split_col: str, label: str) -> dict:
    """Cross-partition overlap using OUR OWN relations, restricted to
    matched entries with a cluster assignment."""
    m = merged.dropna(subset=[split_col, "cluster_rep"]).copy()
    m["_vhvl"] = m["heavy_seq"].fillna("") + "|" + m["light_seq"].fillna("")
    train = m[m[split_col] == "train"]
    test = m[m[split_col] == "test"]

    def overlap(colname):
        tr = set(train[colname].dropna())
        te = set(test[colname].dropna())
        return len(tr & te)

    out = {
        "label": label,
        "n_matched_with_cluster": int(len(m)),
        "cdrh3_cluster_overlap": overlap("cluster_rep"),
        "pdb_overlap": overlap("pdb_id"),
        "antigen_cluster_overlap": overlap("antigen_cluster_id"),
        "exact_vhvl_overlap": overlap("_vhvl"),
    }
    return out


def representativeness_check(merged: pd.DataFrame, split_col: str, label: str) -> dict:
    """Train vs test composition drift on OUR OWN labels (antigen_class,
    antigen_cluster_id, both already precomputed in master_antibodies.csv),
    for matched antigen-bound entries only."""
    m = merged.dropna(subset=[split_col])
    m = m[m["has_antigen"] == True]
    out = {"label": label}
    for part in ["train", "test"]:
        sub = m[m[split_col] == part]
        row = {"n": int(len(sub))}
        vc = sub["antigen_cluster_id"].dropna().value_counts()
        if len(vc) > 0:
            row["antigen_cluster_n_unique"] = int(vc.shape[0])
            row["antigen_cluster_entropy_bits"] = shannon_entropy(vc.values)
            row["antigen_cluster_gini"] = gini_coefficient(vc.values)
        vc2 = sub["antigen_class"].dropna().value_counts()
        if vc2.sum() > 0:
            row["viral_fraction"] = float(vc2.get("viral", 0) / vc2.sum())
        out[part] = row
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits_dir", default="splits_final",
                         help="Directory containing SAbDab2's ab_split.csv and "
                              "abag_split.csv (download from Zenodo record 20083995).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    out_path = os.path.join(work_dir, "tables", "audit_sabdab2_splits.json")

    require_path(master_path, "master_antibodies.csv")
    master = pd.read_csv(master_path)
    log(f"Loaded master_antibodies.csv: {len(master)} rows")

    for col in REQUIRED_MASTER_COLS:
        if col not in master.columns:
            sys.exit(f"ABORT: expected column '{col}' not found in {master_path}")
    log(f"cluster_rep populated: {master['cluster_rep'].notna().sum()} / {len(master)}")
    log(f"antigen_class populated: {master['antigen_class'].notna().sum()} / {len(master)}")

    results = {}
    for fname, split_col, key in [
        ("ab_split.csv", "ab_split", "ab_split"),
        ("abag_split.csv", "ab_ag_split", "ab_ag_split"),
    ]:
        log("=" * 70)
        log(f"{fname}")
        s2 = load_split(args.splits_dir, fname, split_col)
        log(f"Loaded {len(s2)} SAbDab2 entries")
        log(f"{split_col} value counts: {dict(s2[split_col].value_counts())}")

        merged = join_to_our_population(s2, master)
        if len(merged) > len(s2):
            dupe_keys = (merged.groupby(["pdb_key", "h_key", "l_key"])
                         .size().reset_index(name="n"))
            dupe_keys = dupe_keys[dupe_keys["n"] > 1]
            log(f"WARNING: join fanned out ({len(merged)} rows from {len(s2)} input rows) -- "
                f"{len(dupe_keys)} (pdb,h_chain,l_chain) keys matched more than one row in "
                f"master_antibodies.csv (likely multiple 'model_id' entries e.g. NMR ensembles). "
                f"Deduplicating to first match per SAbDab2 INSTANCE before proceeding.")
            merged = merged.drop_duplicates(subset=["INSTANCE"], keep="first")

        n_matched = merged["pdb_id"].notna().sum()
        log(f"Matched to our population: {n_matched} / {len(s2)} "
            f"({100*n_matched/len(s2):.1f}%)")

        unmatched_sample = merged.loc[merged["pdb_id"].isna(), "PDB_ID"].head(10).tolist()
        log(f"Sample of unmatched raw PDB_ID values (first 10): {unmatched_sample}")

        if n_matched == 0:
            sys.exit(f"ABORT: zero matches for {fname} -- the PDB-ID cleaning or the "
                      f"join key is wrong. Do not trust anything below until this is fixed.")

        leak = leakage_check(merged, split_col, key)
        rep = representativeness_check(merged, split_col, key)
        log(f"Leakage check ({key}): {leak}")
        log(f"Representativeness check ({key}): {rep}")

        results[key] = {
            "n_sabdab2_entries": int(len(s2)),
            "n_matched": int(n_matched),
            "match_rate": float(n_matched / len(s2)),
            "leakage_check": leak,
            "representativeness_check": rep,
        }

    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
