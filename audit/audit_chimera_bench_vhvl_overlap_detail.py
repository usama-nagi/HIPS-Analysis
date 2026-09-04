#!/usr/bin/env python3
"""
audit_chimera_bench_vhvl_overlap_detail.py
=============================================
Row-level detail behind Table 2's "Cross-split exact VH/VL pairs" column
(5 / 9 / 5 for Temporal / Epitope-group / Antigen-fold). That count, as
computed by compute_metrics() in audit_chimera_bench_splits.py, is the
number of DISTINCT (heavy_seq, light_seq) sequence-pair VALUES that occur
in more than one split -- i.e. distinct antibody identities by exact
sequence, not rows and not train-test row-pairs. This script recomputes
that same count and, for each one, prints every matched row it came from
(CHIMERA-Bench complex ID, split, PDB, resolution, chain identifiers), so
the "5" or "9" figure is traceable down to specific complexes rather than
taken on faith.

Confirms the counting unit for Table 2's caption: a value of 5 means 5
distinct VH/VL sequences are each shared across 2+ splits -- not 5 rows
and not 5 row-to-row pairings (a single shared sequence appearing in 3
rows across 2 splits still counts once here, as one distinct sequence
spanning multiple splits).

load_master and map_cb_split_to_df are imported directly from
audit_chimera_bench_splits.py rather than redefined here -- see that
script's docstring for why.

Usage
-----
    python scripts/audits/audit_chimera_bench_vhvl_overlap_detail.py \
        --config configs/config.yaml --splits_dir <chimera_bench_splits_dir>

Reads:
    <work_dir>/tables/master_antibodies.csv
    <chimera_bench_splits_dir>/{temporal,epitope_group,antigen_fold}.json

Writes:
    <work_dir>/tables/chimera_bench_vhvl_overlap_detail.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, log
from audit_chimera_bench_splits import load_master, load_cb_splits, map_cb_split_to_df


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

    cb_splits = load_cb_splits(args.splits_dir)
    results = {}
    for key, cb_dict in cb_splits.items():
        row_to_split, row_to_cbid = map_cb_split_to_df(df, cb_dict)
        sub = df.loc[list(row_to_split.keys())].copy()
        sub["_split"] = pd.Series(row_to_split)
        sub["_cbid"] = pd.Series(row_to_cbid)
        sub["_vhvl_key"] = sub["heavy_seq"].astype(str) + "||" + sub["light_seq"].astype(str)

        dup_key_splits = sub.groupby("_vhvl_key")["_split"].nunique()
        dup_keys = dup_key_splits[dup_key_splits > 1].index

        log(f"=== {key}: {len(dup_keys)} distinct VH/VL sequences spanning >1 split ===")
        entries_for_key = []
        for vhvl_key in dup_keys:
            rows = sub[sub["_vhvl_key"] == vhvl_key]
            occurrences = []
            for _, r in rows.iterrows():
                occurrences.append({
                    "cb_id": r["_cbid"],
                    "split": r["_split"],
                    "pdb_id": r["pdb_id"],
                    "resolution": r.get("resolution", None),
                    "h_chain": r["h_chain"],
                    "l_chain": r["l_chain"],
                })
            entries_for_key.append({"n_occurrences": len(rows), "occurrences": occurrences})
            log(f"  VH/VL pair appears in {len(rows)} rows across splits:")
            for occ in occurrences:
                log(f"    cb_id={occ['cb_id']}  split={occ['split']}  pdb={occ['pdb_id']}  "
                    f"resolution={occ['resolution']}  h_chain={occ['h_chain']}  l_chain={occ['l_chain']}")

        results[key] = {
            "n_distinct_vhvl_pairs_spanning_multiple_splits": int(len(dup_keys)),
            "detail": entries_for_key,
        }

    out_path = os.path.join(work_dir, "tables", "chimera_bench_vhvl_overlap_detail.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
