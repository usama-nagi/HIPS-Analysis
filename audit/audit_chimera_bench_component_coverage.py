#!/usr/bin/env python3
"""
audit_chimera_bench_component_coverage.py
============================================
Independent integrity check on the "own components cut across CHIMERA-
Bench splits" cross-check that audit_chimera_bench_splits.py already
reports inline (Part B). Recomputes it as a standalone script with its
own explicit denominators, so the figure can be verified without trusting
a single combined script to get its own bookkeeping right: for each
CHIMERA-Bench split file, how many of the matched entries actually
received a _full_component value at all (should be all of them -- a
gap here would mean matched entries fell outside every connected-
component relation, which should not happen since cluster_rep and
pdb_id are populated for every row), and, restricted to entries that
did, how many of our independently-computed connected components (CDR-H3
cluster + PDB + antigen cluster) are cut across more than one of
CHIMERA-Bench's own splits, and how many entries that affects.

build_components, load_master, and map_cb_split_to_df are imported
directly from audit_chimera_bench_splits.py rather than redefined here --
see that script's docstring for why.

Usage
-----
    python scripts/audits/audit_chimera_bench_component_coverage.py \
        --config configs/config.yaml --splits_dir <chimera_bench_splits_dir>

Reads:
    <work_dir>/tables/master_antibodies.csv
    <chimera_bench_splits_dir>/{temporal,epitope_group,antigen_fold}.json

Writes:
    <work_dir>/tables/chimera_bench_component_coverage.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, log
from audit_chimera_bench_splits import (
    build_components, load_master, load_cb_splits, map_cb_split_to_df,
)

FULL_RELATIONS = ["cluster_rep", "pdb_id", "antigen_cluster_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits_dir", required=True,
                         help="Directory containing CHIMERA-Bench's temporal.json, "
                              "epitope_group.json, and antigen_fold.json.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    df = load_master(work_dir)
    log(f"Loaded master_antibodies.csv: {len(df)} entries")

    df["_full_component"] = build_components(df, FULL_RELATIONS)

    cb_splits = load_cb_splits(args.splits_dir)
    results = {}
    for key, cb_dict in cb_splits.items():
        row_to_split, _ = map_cb_split_to_df(df, cb_dict)
        n_matched = len(row_to_split)
        sub = df.loc[list(row_to_split.keys())].copy()
        sub["_split"] = pd.Series(row_to_split)

        n_with_component = int(sub["_full_component"].notna().sum())

        comp_split_counts = sub.groupby("_full_component")["_split"].nunique()
        split_components = comp_split_counts[comp_split_counts > 1].index
        n_components_split = len(split_components)
        n_components_total = len(comp_split_counts)
        entries_affected = int(sub[sub["_full_component"].isin(split_components)].shape[0])

        results[key] = {
            "n_matched": n_matched,
            "n_with_component": n_with_component,
            "n_components_total": int(n_components_total),
            "n_components_cut_across_splits": int(n_components_split),
            "entries_affected": entries_affected,
            "entries_affected_pct_of_matched": round(100 * entries_affected / n_matched, 1),
        }

        log(f"=== {key} ===")
        log(f"  matched entries: {n_matched}  |  with a _full_component value: {n_with_component}")
        log(f"  components: {n_components_total}  |  cut across >1 split: {n_components_split}")
        log(f"  entries affected: {entries_affected} "
            f"({100*entries_affected/n_matched:.1f}% of matched entries)")

    out_path = os.path.join(work_dir, "tables", "chimera_bench_component_coverage.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
