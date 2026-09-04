#!/usr/bin/env python3
"""
audit_chimera_bench_giant_component_composition.py
=====================================================
Follow-up to Part A of audit_chimera_bench_splits.py. That script shows a
standard greedy 80/10/10 partitioner still "succeeds" by entry count even
when a giant connected component (all three relations: CDR-H3 cluster,
PDB identifier, antigen cluster) has to be assigned intact to one split.
This script answers the natural next question: succeeding by entry count
doesn't mean succeeding by composition -- what does the resulting
per-split antigen_class distribution actually look like, and in
particular how much does the viral-antigen fraction diverge across
splits? This is the source of the paper's "Connected-component
splitting" paragraph's 45.2% vs. 10.8--14.5% viral-fraction figures.

build_components and greedy_partition are imported directly from
audit_chimera_bench_splits.py rather than redefined here, so this script
cannot silently diverge from that script's definition of "connected
component" or "greedy partition" -- see that script's own docstring for
the full rationale. (An earlier ad-hoc version of this check kept a
local copy of greedy_partition with a different return signature than
the one in the main audit script; importing removes that risk
structurally rather than relying on two copies being kept in sync by
hand.)

Usage
-----
    python scripts/audits/audit_chimera_bench_giant_component_composition.py \
        --config configs/config.yaml

Reads:
    <work_dir>/tables/master_antibodies.csv

Writes:
    <work_dir>/tables/chimera_bench_giant_component_composition.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, log
from audit_chimera_bench_splits import build_components, greedy_partition, load_master

FULL_RELATIONS = ["cluster_rep", "pdb_id", "antigen_cluster_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    df = load_master(work_dir)
    log(f"Loaded master_antibodies.csv: {len(df)} entries")

    if "antigen_class" not in df.columns:
        sys.exit("[SCHEMA MISMATCH] master_antibodies.csv is missing antigen_class. "
                  "Run migrate_master_csv.py --only antigen_class first.")

    df["_comp"] = build_components(df, FULL_RELATIONS)
    split_series, achieved = greedy_partition(df, "_comp")
    df["_split"] = split_series

    comp_counts = df["_comp"].value_counts()
    giant_id = comp_counts.index[0]
    giant_size = int(comp_counts.iloc[0])
    giant_split = df.loc[df["_comp"] == giant_id, "_split"].iloc[0]
    log(f"Giant component ({giant_size} entries, {100*giant_size/len(df):.1f}% of corpus) "
        f"assigned to: {giant_split}")

    per_split = {}
    for s in ["train", "val", "test"]:
        sub = df[df["_split"] == s]
        dist = sub["antigen_class"].value_counts(normalize=True).mul(100).round(1)
        per_split[s] = {
            "n": int(len(sub)),
            "antigen_class_pct": dist.to_dict(),
            "viral_pct": float(dist.get("viral", 0.0)),
        }
        log(f"\n{s} ({len(sub)} entries) antigen_class distribution:")
        log(dist.to_string())

    result = {
        "giant_component_id": str(giant_id),
        "giant_component_size": giant_size,
        "giant_component_pct_of_corpus": round(100 * giant_size / len(df), 1),
        "giant_component_assigned_split": str(giant_split),
        "per_split": per_split,
    }

    out_path = os.path.join(work_dir, "tables", "chimera_bench_giant_component_composition.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
