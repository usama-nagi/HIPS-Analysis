"""
rq3_dedup_threshold_sensitivity.py

Extension to RQ3. The paper's benchmark-design recommendations rest on a
single CDR-H3 clustering threshold (90% identity), chosen as the standard
MMseqs2 near-duplicate-detection convention. This script asks the natural
follow-up question: how much would the headline dedup numbers change at a
different threshold? It re-runs MMseqs2 clustering at several identity
thresholds and recomputes the same before/after diversity snapshot RQ3
Section A already computes, so the sensitivity numbers are produced by
the exact same metric-computation code path -- not a re-derivation that
could silently diverge.

This does NOT replace rq3_redundancy_and_recommendations.py's own 90%
threshold result, which remains the paper's primary, citable number
(and whose underlying cluster assignments are already reused throughout
RQ1/RQ3 for other purposes, e.g. the SAbDab clonotype proxy in RQ2). This
script is a robustness check reported alongside it.

CDR-H3 extraction and MMseqs2 clustering are imported directly from
rq1_sequence_structural_bias.py rather than kept as local copies, so this
script cannot diverge from that file's clustering logic -- both use
exactly the same FASTA-ID handling and exact-duplicate-collapse behavior
(see extract_cdrh3_sequences / run_mmseqs2_cluster in that file for the
full rationale). One consequence of importing the real functions: they
raise a hard [INTEGRITY ERROR] if any sequence's cluster assignment
cannot be resolved, rather than silently warning and dropping rows, which
is intentional -- see those functions' own docstrings. If this script
raises that error on your data, it means something about your
filename_stem values or your MMseqs2 output needs investigating, not a
condition to fall back past.

Usage:
    python rq3_dedup_threshold_sensitivity.py --config config.yaml --thresholds 0.80 0.85 0.90 0.95

Reads:
    <work_dir>/tables/master_antibodies.csv

Writes:
    <work_dir>/tables/rq3_dedup_threshold_sensitivity.json
    <work_dir>/tmp_mmseqs_cdrh3_sensitivity_<threshold>/   (intermediate MMseqs2 files per threshold)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Imports the shared clustering functions rather than keeping local
# copies, so this script cannot drift out of sync with
# rq1_sequence_structural_bias.py's clustering logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_sequence_structural_bias import extract_cdrh3_sequences, run_mmseqs2_cluster


def shannon_entropy(counts, base=2.0):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * (np.log(p) / np.log(base))).sum())


def normalized_entropy(counts, base=2.0):
    counts = np.asarray(counts, dtype=np.float64)
    n_categories = int((counts > 0).sum())
    if n_categories <= 1:
        return 0.0
    h = shannon_entropy(counts, base=base)
    h_max = np.log(n_categories) / np.log(base)
    return float(h / h_max) if h_max > 0 else 0.0


def gini_coefficient(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts >= 0]
    if counts.sum() == 0 or len(counts) == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1) * cum[-1]) / (n * cum[-1])
    return float(gini)


def pick_cluster_representatives(master_df, member_to_rep):
    """member_to_rep is now {filename_stem: cluster_rep_filename_stem} for
    EVERY filename_stem extracted for clustering -- run_mmseqs2_cluster()
    itself raises an [INTEGRITY ERROR] if any are missing, so no .isna()
    drop path is needed or used here. This mirrors
    rq1_sequence_structural_bias.py's run_section_b, which removed the old
    dropna() path for the same reason."""
    df = master_df.copy()
    df["cluster_rep"] = df["filename_stem"].map(member_to_rep)
    n_unmatched = df["cluster_rep"].isna().sum()
    if n_unmatched > 0:
        raise RuntimeError(
            f"[INTEGRITY ERROR] {n_unmatched} rows in master_df had no cluster "
            f"assignment after mapping through an integrity-checked member_to_rep. "
            f"This should be impossible -- investigate before trusting this run."
        )
    df = df.sort_values(by=["cluster_rep", "has_antigen", "pdb_id"], ascending=[True, False, True])
    return df.drop_duplicates(subset=["cluster_rep"], keep="first")


def compute_diversity_snapshot(df):
    out = {"n_antibodies": int(len(df))}

    if "cdr3_len_actual" in df.columns:
        lens = df["cdr3_len_actual"].dropna().astype(int)
        len_counts = lens.value_counts()
        out["length_entropy_bits"] = shannon_entropy(len_counts.values)
        out["length_normalized_entropy"] = normalized_entropy(len_counts.values)
        out["length_gini"] = gini_coefficient(len_counts.values)
        out["length_mean"] = float(lens.mean()) if len(lens) else None

    if "heavy_subclass" in df.columns:
        vc = df["heavy_subclass"].fillna("UNKNOWN").value_counts()
        out["heavy_germline_entropy_bits"] = shannon_entropy(vc.values)
        out["heavy_germline_gini"] = gini_coefficient(vc.values)
        out["heavy_germline_n_unique"] = int(vc.shape[0])

    if "antigen_cluster_id" in df.columns and "has_antigen" in df.columns:
        ag = df.loc[df["has_antigen"] == True, "antigen_cluster_id"].dropna()
        if len(ag) > 0:
            vc = ag.value_counts()
            out["antigen_cluster_entropy_bits"] = shannon_entropy(vc.values)
            out["antigen_cluster_gini"] = gini_coefficient(vc.values)
            out["antigen_cluster_n_unique"] = int(vc.shape[0])

    if "is_therapeutic" in df.columns:
        out["therapeutic_fraction"] = float(df["is_therapeutic"].mean())

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95],
                     help="MMseqs2 min-seq-id thresholds to sweep. 0.90 should match the "
                          "paper's primary result (rq3_before_after_dedup.json) -- if it "
                          "doesn't match closely, something has diverged between this "
                          "script's clustering call and the original Section B call "
                          "and should be investigated before trusting the sweep.")
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--n_threads", type=int, default=16)
    ap.add_argument("--master_csv", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    work_dir = config["paths"]["work_dir"]
    master_path = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")
    out_path = os.path.join(work_dir, "tables", "rq3_dedup_threshold_sensitivity.json")

    if not os.path.exists(master_path):
        sys.exit(f"[FILE NOT FOUND] {master_path}")

    master_df = pd.read_csv(master_path)
    print(f"[INFO] Loaded {len(master_df)} rows from {master_path}")

    print("[INFO] Isolating CDR-H3 sequences once (reused across all thresholds)...")
    cdrh3_seqs, index_to_filename_stem = extract_cdrh3_sequences(master_df)
    print(f"[INFO] Extracted {len(cdrh3_seqs)} CDR-H3 sequences "
          f"(out of {len(master_df)} input rows)")
    if len(cdrh3_seqs) != len(master_df):
        print(f"[NOTE] {len(master_df) - len(cdrh3_seqs)} rows did not yield a usable "
              f"CDR-H3 sequence (load error, missing mask, or H3 too short) and are "
              f"excluded from clustering at every threshold below. This exclusion is "
              f"threshold-independent -- it happens before any MMseqs2 run.")

    before = compute_diversity_snapshot(master_df)
    print(f"[INFO] BEFORE dedup (threshold-independent): {before}")

    results = {"before": before, "by_threshold": {}}

    for threshold in args.thresholds:
        print(f"\n{'='*70}\nThreshold: {threshold}\n{'='*70}")
        tmp_dir = os.path.join(work_dir, f"tmp_mmseqs_cdrh3_sensitivity_{threshold}")
        member_to_rep = run_mmseqs2_cluster(
            cdrh3_seqs, index_to_filename_stem, tmp_dir, threshold, args.coverage, args.n_threads
        )

        n_clusters = len(set(member_to_rep.values()))
        dedup_df = pick_cluster_representatives(master_df, member_to_rep)
        after = compute_diversity_snapshot(dedup_df)

        delta = {}
        for k in before:
            if k in after and isinstance(before[k], (int, float)) and isinstance(after[k], (int, float)):
                if k == "n_antibodies":
                    delta[k] = {"before": before[k], "after": after[k],
                                "reduction_fraction": 1 - (after[k] / max(before[k], 1))}
                else:
                    abs_delta = after[k] - before[k]
                    rel_delta = abs_delta / abs(before[k]) if before[k] not in (0, None) else None
                    delta[k] = {"before": before[k], "after": after[k],
                                "absolute_delta": abs_delta, "relative_delta": rel_delta}

        results["by_threshold"][str(threshold)] = {
            "n_clusters": n_clusters,
            "n_sequences_clustered": len(cdrh3_seqs),
            "compression_ratio": len(cdrh3_seqs) / max(n_clusters, 1),
            "after": after,
            "delta": delta,
        }
        print(f"[INFO] threshold={threshold}: n_clusters={n_clusters}, "
              f"n_after_dedup={after['n_antibodies']}, "
              f"reduction_fraction={delta['n_antibodies']['reduction_fraction']:.3f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Wrote {out_path}")

    print("\nSummary across thresholds:")
    print(f"{'threshold':<12}{'n_clusters':<12}{'n_after_dedup':<16}{'reduction_frac':<16}{'antigen_cluster_gini_after':<28}")
    for t in args.thresholds:
        r = results["by_threshold"][str(t)]
        ag_gini = r["after"].get("antigen_cluster_gini", float("nan"))
        print(f"{t:<12}{r['n_clusters']:<12}{r['after']['n_antibodies']:<16}"
              f"{r['delta']['n_antibodies']['reduction_fraction']:<16.3f}{ag_gini:<28.3f}")


if __name__ == "__main__":
    main()