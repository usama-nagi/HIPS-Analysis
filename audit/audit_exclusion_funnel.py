#!/usr/bin/env python3
"""
audit_exclusion_funnel.py

Reads master_antibodies.csv directly (one row per chain-pair/antigen/model
combination -- confirmed against 00_build_dataset.py's .pt filename
convention: f"{pdb}_{h}{l}_ag{ag}_m{model}.pt"). Produces four counts:

  1. rows                 = total master_antibodies.csv rows (should be 20,037)
  2. coordinate_units      = unique (pdb_id, h_chain, l_chain, model_id) --
                             collapses duplicate rows that differ only in
                             which antigen chain was requested
  3. vh_vl_pairs           = unique (heavy_seq, light_seq) sequence pairs --
                             collapses re-depositions of the identical
                             antibody under different PDB entries
  4. cross_pdb_clones      = VH/VL pairs that appear under >1 distinct PDB id

Also recomputes the exact-CDR-H3-sharing fraction (paper's 89.9% claim) at
both the row level (reproduces the existing number, sanity check) and the
coordinate-unit level (the new number the paper needs).

Usage
-----
    python scripts/audit_exclusion_funnel.py --config configs/config.yaml
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

SENTINEL = "__MISSING__"


def _key(df: pd.DataFrame, cols):
    """Build a tuple-key Series over `cols`, filling NaN with a sentinel so
    groupby doesn't silently drop rows with missing values in any column."""
    return list(zip(*[df[c].fillna(SENTINEL).astype(str) for c in cols]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--master_csv", default=None,
                         help="Override path to master_antibodies.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    master_path = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")
    df = pd.read_csv(master_path, low_memory=False)
    log(f"Loaded {master_path}: {len(df)} rows, {len(df.columns)} columns")

    required_cols = ["pdb_id", "heavy_seq", "light_seq"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[SCHEMA MISMATCH] master_antibodies.csv is missing {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # h_chain/l_chain/model_id_raw only exist for rows that matched the TSV
    # join in 00_build_dataset.py (see build_report.json's n_unmatched_pt_to_tsv).
    has_chain_cols = "h_chain" in df.columns and "l_chain" in df.columns
    model_col = "model_id_raw" if "model_id_raw" in df.columns else (
        "model_id" if "model_id" in df.columns else None)

    n_rows = len(df)
    n_unmatched = int(df["h_chain"].isna().sum()) if has_chain_cols else None

    report = {"n_rows": n_rows}

    # 1. Coordinate units: unique (pdb_id, h_chain, l_chain, model_id)
    if has_chain_cols and model_col:
        coord_key = _key(df, ["pdb_id", "h_chain", "l_chain", model_col])
        df["_coord_unit"] = coord_key
        n_coord_units = df["_coord_unit"].nunique()
        # how many rows collapse per unit, i.e. same coordinate unit repeating
        # across distinct antigen chains
        unit_sizes = df.groupby("_coord_unit").size()
        n_units_with_multi_antigen = int((unit_sizes > 1).sum())
        report["coordinate_units"] = {
            "n_unique": int(n_coord_units),
            "n_rows_collapsed": int(n_rows - n_coord_units),
            "n_units_spanning_multiple_antigen_chains": n_units_with_multi_antigen,
            "note": "Unique (pdb_id, h_chain, l_chain, model_id); collapses "
                    "rows that differ only in requested antigen chain.",
        }
        log(f"Coordinate units: {n_coord_units} unique / {n_rows} rows "
            f"({n_units_with_multi_antigen} span >1 antigen chain)")
    else:
        report["coordinate_units"] = {
            "status": "SKIPPED",
            "reason": f"h_chain/l_chain/model_id not present as columns "
                      f"(found: {list(df.columns)}). Check the TSV-join match "
                      f"rate in build_report.json first.",
        }
        log("WARNING: coordinate-unit count skipped -- h_chain/l_chain/model_id "
            "columns not found. See build_report.json.")

    # 2. Unique VH/VL pairs (collapses re-depositions of the same Ab)
    df["_vh_vl_key"] = _key(df, ["heavy_seq", "light_seq"])
    n_vh_vl = df["_vh_vl_key"].nunique()
    report["vh_vl_pairs"] = {
        "n_unique": int(n_vh_vl),
        "n_rows_collapsed": int(n_rows - n_vh_vl),
        "note": "Unique (heavy_seq, light_seq) full sequence pairs.",
    }
    log(f"Unique VH/VL pairs: {n_vh_vl} / {n_rows} rows")

    # 3. Cross-PDB clones: same VH/VL pair under >1 distinct pdb_id 
    pdb_counts_per_pair = df.groupby("_vh_vl_key")["pdb_id"].nunique()
    cross_pdb_pairs = pdb_counts_per_pair[pdb_counts_per_pair > 1]
    n_cross_pdb_pairs = len(cross_pdb_pairs)
    n_rows_in_cross_pdb_pairs = int(df["_vh_vl_key"].isin(cross_pdb_pairs.index).sum())
    report["cross_pdb_clones"] = {
        "n_vh_vl_pairs_spanning_multiple_pdbs": int(n_cross_pdb_pairs),
        "n_rows_involved": n_rows_in_cross_pdb_pairs,
        "max_pdbs_for_one_pair": int(pdb_counts_per_pair.max()) if len(pdb_counts_per_pair) else 0,
        "note": "VH/VL pairs (identical heavy_seq+light_seq) appearing under "
                "more than one distinct pdb_id -- the same antibody solved "
                "in multiple independent depositions.",
    }
    log(f"Cross-PDB clones: {n_cross_pdb_pairs} VH/VL pairs span multiple PDB "
        f"IDs ({n_rows_in_cross_pdb_pairs} rows total)")

    # 4. Exact-CDR-H3-sharing recompute 
    if "h3_seq" in df.columns:
        # row level -- should reproduce the existing 89.9% as a sanity check
        h3_row_counts = df["h3_seq"].value_counts()
        n_shared_rows = int((df["h3_seq"].map(h3_row_counts) > 1).sum())
        row_level_pct = 100.0 * n_shared_rows / n_rows

        # coordinate-unit level -- one h3_seq per unique coordinate unit
        if has_chain_cols and model_col:
            unit_h3 = df.drop_duplicates("_coord_unit")["h3_seq"]
            unit_h3_counts = unit_h3.value_counts()
            n_shared_units = int((unit_h3.map(unit_h3_counts) > 1).sum())
            unit_level_pct = 100.0 * n_shared_units / len(unit_h3)
        else:
            n_shared_units = unit_level_pct = None

        # VH/VL-pair level -- one h3_seq per unique antibody
        pair_h3 = df.drop_duplicates("_vh_vl_key")["h3_seq"]
        pair_h3_counts = pair_h3.value_counts()
        n_shared_pairs = int((pair_h3.map(pair_h3_counts) > 1).sum())
        pair_level_pct = 100.0 * n_shared_pairs / len(pair_h3)

        report["exact_h3_sharing"] = {
            "row_level_pct": round(row_level_pct, 2),
            "row_level_n_shared": n_shared_rows,
            "row_level_n_total": n_rows,
            "coordinate_unit_level_pct": round(unit_level_pct, 2) if unit_level_pct is not None else None,
            "coordinate_unit_level_n_shared": n_shared_units,
            "vh_vl_pair_level_pct": round(pair_level_pct, 2),
            "vh_vl_pair_level_n_shared": n_shared_pairs,
            "vh_vl_pair_level_n_total": len(pair_h3),
            "note": "row_level should reproduce the paper's existing 89.9% "
                    "(sanity check on this script). coordinate_unit_level and "
                    "vh_vl_pair_level are the new, collapsed-duplicate numbers "
                    "for C2.",
        }
        log(f"Exact H3 sharing -- row level: {row_level_pct:.2f}% (sanity check "
            f"vs. paper's 89.9%), coordinate-unit level: "
            f"{unit_level_pct:.2f}% (new)" if unit_level_pct is not None else
            f"Exact H3 sharing -- row level: {row_level_pct:.2f}%")
    else:
        report["exact_h3_sharing"] = {
            "status": "SKIPPED",
            "reason": "h3_seq column not present. Run: "
                      "python migrate_master_csv.py --config configs/config.yaml --only h3_seq",
        }
        log("WARNING: H3-sharing recompute skipped -- h3_seq column missing. "
            "Run migrate_master_csv.py --only h3_seq first, then re-run this script.")

    out_path = os.path.join(work_dir, "tables", "row_entry_accounting.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {out_path}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
