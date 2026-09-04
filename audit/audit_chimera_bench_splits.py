#!/usr/bin/env python3
"""
audit_chimera_bench_splits.py
==============================
Two related analyses, sharing the same connected-component machinery:

  Part A -- Own-corpus connected-component splitting (feeds the paper's
  "Connected-component splitting" paragraph). Builds connected components
  over our own 19,848-entry corpus under three progressively richer edge
  relations (CDR-H3 cluster only; + PDB identifier; + antigen-cluster
  linking), greedily bin-packs components into an 80/10/10 train/val/test
  split by entry count, and reports cross-split leakage metrics (PDB,
  antigen-cluster, CDR-H3-cluster, and exact-VH/VL overlap) for each
  configuration.

  Part B -- External audit: CHIMERA-Bench. Maps CHIMERA-Bench's own
  published splits (temporal, epitope-group, antigen-fold) onto our
  schema by PDB identifier, chain identifiers, and antigen chain, then
  runs the same leakage-metric computation against those real splits
  (feeds the paper's "External audit: CHIMERA-Bench" paragraph and
  Table 2's "Cross-split exact VH/VL pairs" column) and cross-checks
  whether CHIMERA-Bench's splits cut across our own independently
  computed connected components.

UnionFind / build_components / greedy_partition / compute_metrics /
parse_cb_id / map_cb_split_to_df are defined here and imported directly
by audit_chimera_bench_giant_component_composition.py,
audit_chimera_bench_component_coverage.py, and
audit_chimera_bench_vhvl_overlap_detail.py, rather than each keeping a
local copy -- so all four scripts share exactly one definition of
"connected component" and "CHIMERA-Bench split mapping" and cannot drift
apart from each other.

Requires Stage 4's cluster_rep and antigen_cluster_id columns on
master_antibodies.csv.

Usage
-----
    python scripts/audits/audit_chimera_bench_splits.py --config configs/config.yaml \
        --splits_dir <chimera_bench_splits_dir>

Reads:
    <work_dir>/tables/master_antibodies.csv
    <chimera_bench_splits_dir>/temporal.json
    <chimera_bench_splits_dir>/epitope_group.json
    <chimera_bench_splits_dir>/antigen_fold.json
        (CHIMERA-Bench's own published split files -- not part of this
        repository; each is a dict of {split_name: [complex_id, ...]}
        with complex_id formatted "pdb_hchain_lchain_agchain")

Writes:
    <work_dir>/tables/chimera_bench_splits_audit.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

REQUIRED_MASTER_COLS = [
    "pdb_id", "h_chain", "l_chain", "heavy_seq", "light_seq",
    "cluster_rep", "antigen_cluster_id", "antigen_chain_str",
]

OWN_CORPUS_CONFIGS = {
    "cdrh3_only": ["cluster_rep"],
    "cdrh3_plus_pdb": ["cluster_rep", "pdb_id"],
    "cdrh3_plus_pdb_plus_antigen": ["cluster_rep", "pdb_id", "antigen_cluster_id"],
}

CB_SPLIT_FILES = {
    "temporal": "temporal.json",
    "epitope_group": "epitope_group.json",
    "antigen_fold": "antigen_fold.json",
}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_components(df: pd.DataFrame, relation_cols) -> np.ndarray:
    """Union-find connected components over df's row index, linking any
    two rows that share a non-null value in any of relation_cols."""
    uf = UnionFind(len(df))
    for col in relation_cols:
        for val, rows in df.groupby(col).indices.items():
            if pd.isna(val) or len(rows) < 2:
                continue
            first = rows[0]
            for r in rows[1:]:
                uf.union(first, r)
    return np.array([uf.find(i) for i in range(len(df))])


def greedy_partition(df: pd.DataFrame, comp_col: str,
                      targets=None) -> pd.Series:
    """Greedily assigns whole connected components (largest first) to
    whichever of train/val/test currently has the most remaining budget
    against an 80/10/10 entry-count target. A single giant component can
    still push its assigned split over budget -- that is the point: this
    is what shows a standard partitioner "succeeding" by entry count while
    still concentrating composition (see Part A below)."""
    if targets is None:
        targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    n = len(df)
    target_sizes = {k: v * n for k, v in targets.items()}
    current = {k: 0 for k in targets}
    comp_sizes = df.groupby(comp_col).size().sort_values(ascending=False)
    assignment = {}
    for comp_id, size in comp_sizes.items():
        best_split = max(current, key=lambda k: target_sizes[k] - current[k])
        assignment[comp_id] = best_split
        current[best_split] += size
    return df[comp_col].map(assignment), current


def compute_metrics(df: pd.DataFrame, split_series: pd.Series, label: str,
                     giant_component_frac=None) -> dict:
    """Cross-split leakage metrics for a given split assignment: how many
    distinct PDB identifiers, antigen clusters, CDR-H3 clusters, and exact
    VH/VL sequence pairs occur in more than one split. Returns a dict
    (also logged) rather than printing only, so every number here is
    traceable to this function's JSON output, not just a console log."""
    d = df.copy()
    d["_split"] = split_series.values
    sizes = d["_split"].value_counts()
    total = len(d)
    split_sizes = {s: {"n": int(sizes.get(s, 0)),
                        "pct": round(100 * sizes.get(s, 0) / total, 1)}
                   for s in ["train", "val", "test"]}

    def overlap_count(col):
        g = d.dropna(subset=[col]).groupby(col)["_split"].nunique()
        return int((g > 1).sum())

    pdb_overlap = overlap_count("pdb_id")
    ag_overlap = overlap_count("antigen_cluster_id")
    h3_overlap = overlap_count("cluster_rep")

    d["_vhvl_key"] = d["heavy_seq"].astype(str) + "||" + d["light_seq"].astype(str)
    vhvl_overlap = overlap_count("_vhvl_key")

    result = {
        "label": label,
        "split_sizes": split_sizes,
        "pdb_ids_spanning_multiple_splits": pdb_overlap,
        "antigen_clusters_spanning_multiple_splits": ag_overlap,
        "cdrh3_clusters_spanning_multiple_splits": h3_overlap,
        "exact_vhvl_pairs_spanning_multiple_splits": vhvl_overlap,
    }
    if giant_component_frac is not None:
        result["largest_component_pct_of_corpus"] = round(giant_component_frac, 1)

    log(f"=== {label} ===")
    for s, v in split_sizes.items():
        log(f"  {s}: {v['n']} ({v['pct']}%)")
    log(f"  PDB IDs spanning >1 split: {pdb_overlap}")
    log(f"  antigen clusters spanning >1 split: {ag_overlap}")
    log(f"  CDR-H3 clusters (90% id) spanning >1 split: {h3_overlap}")
    log(f"  exact VH/VL pairs spanning >1 split: {vhvl_overlap}")
    if giant_component_frac is not None:
        log(f"  entries in largest single component: {giant_component_frac:.1f}% of corpus")
    return result


def parse_cb_id(cb_id: str):
    """CHIMERA-Bench complex IDs are 'pdb_hchain_lchain_agchain'. Returns
    (pdb_lower, h_chain, l_chain, ag_chain) or None if the ID doesn't
    match this shape."""
    parts = cb_id.split("_")
    if len(parts) != 4:
        return None
    pdb, h, l, ag = parts
    return pdb.lower(), h, l, ag


def map_cb_split_to_df(df: pd.DataFrame, cb_dict: dict):
    """Maps a CHIMERA-Bench split dict ({split_name: [complex_id, ...]})
    onto df's row index by PDB identifier, chain identifiers, and antigen
    chain membership. Returns (row_to_split, row_to_cbid) dicts keyed by
    df row index. df must already have pdb_lower and ag_chains columns
    (see main() below)."""
    row_to_split = {}
    row_to_cbid = {}
    for split_name, id_list in cb_dict.items():
        for cb_id in id_list:
            parsed = parse_cb_id(cb_id)
            if parsed is None:
                continue
            pdb, h, l, ag = parsed
            hits = df[(df["pdb_lower"] == pdb) & (df["h_chain"] == h) & (df["l_chain"] == l)]
            hits = hits[hits["ag_chains"].apply(lambda s: ag in s)]
            for i in hits.index:
                row_to_split[i] = split_name
                row_to_cbid[i] = cb_id
    return row_to_split, row_to_cbid


def add_cb_join_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds pdb_lower and ag_chains columns used by map_cb_split_to_df.
    Shared here so every script that calls map_cb_split_to_df prepares
    df identically."""
    df = df.copy()
    df["pdb_lower"] = df["pdb_id"].str.lower()
    df["ag_chains"] = df["antigen_chain_str"].fillna("").apply(
        lambda s: set(c.strip() for c in s.split("_") if c.strip())
    )
    return df


def load_master(work_dir: str) -> pd.DataFrame:
    """Loads master_antibodies.csv, validates required columns, and adds
    the CHIMERA-Bench join columns. Shared entry point for all four
    audit_chimera_bench_*.py scripts."""
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")
    df = pd.read_csv(master_path)
    missing = [c for c in REQUIRED_MASTER_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[SCHEMA MISMATCH] master_antibodies.csv missing {missing}. "
                  f"cluster_rep and antigen_cluster_id must be populated first "
                  f"-- see migrate_master_csv.py --only cluster_rep.")
    return add_cb_join_columns(df)


def load_cb_splits(splits_dir: str) -> dict:
    """Loads all three CHIMERA-Bench split files. Exits with a clear
    message if the directory hasn't been populated with the external
    download."""
    out = {}
    for key, fname in CB_SPLIT_FILES.items():
        path = os.path.join(splits_dir, fname)
        require_path(path, f"CHIMERA-Bench split file (external download, see README)")
        with open(path) as f:
            out[key] = json.load(f)
    return out


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
    n = len(df)
    log(f"Loaded master_antibodies.csv: {n} entries")

    results = {"own_corpus": {}, "chimera_bench_external_audit": {}}

    # ---- Part A: own-corpus connected-component splitting ----
    own_components = {}
    for label, relations in OWN_CORPUS_CONFIGS.items():
        comp = build_components(df, relations)
        df["_comp"] = comp
        own_components[label] = comp.copy()
        largest_frac = 100 * df["_comp"].value_counts().iloc[0] / n
        n_components = df["_comp"].nunique()
        split_series, achieved = greedy_partition(df, "_comp")
        metrics = compute_metrics(df, split_series, label, giant_component_frac=largest_frac)
        metrics["relations"] = relations
        metrics["n_components"] = int(n_components)
        results["own_corpus"][label] = metrics

    # Keep the full-3-relation component assignment for the CHIMERA-Bench
    # cross-check in Part B.
    df["_full_component"] = own_components["cdrh3_plus_pdb_plus_antigen"]

    # ---- Part B: external audit against CHIMERA-Bench's real splits ----
    cb_splits = load_cb_splits(args.splits_dir)
    for key, cb_dict in cb_splits.items():
        row_to_split, _ = map_cb_split_to_df(df, cb_dict)
        n_matched = len(row_to_split)
        sub = df.loc[list(row_to_split.keys())].copy()
        sub["_split"] = pd.Series(row_to_split)
        metrics = compute_metrics(sub, sub["_split"], f"chimera_bench_{key}")
        metrics["n_matched"] = n_matched

        comp_split_counts = sub.groupby("_full_component")["_split"].nunique()
        n_components_cut = int((comp_split_counts > 1).sum())
        n_components_total = int(comp_split_counts.shape[0])
        entries_in_cut_components = int(sub[sub["_full_component"].isin(
            comp_split_counts[comp_split_counts > 1].index)].shape[0])
        metrics["own_components_cut_across_splits"] = n_components_cut
        metrics["own_components_total_among_matched"] = n_components_total
        metrics["entries_in_cut_components"] = entries_in_cut_components
        log(f"  [cross-check vs. own connected components] "
            f"{n_components_cut}/{n_components_total} of our components are "
            f"cut across >1 of CHIMERA-Bench's {key} splits, affecting "
            f"{entries_in_cut_components} entries")

        results["chimera_bench_external_audit"][key] = metrics

    out_path = os.path.join(work_dir, "tables", "chimera_bench_splits_audit.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
