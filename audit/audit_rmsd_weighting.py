#!/usr/bin/env python3
"""
audit_rmsd_weighting.py
=======================================
Recomputes CDR-H3 C-alpha RMSD over every non-singleton cluster, including
the two largest (649 and 373 members), which a default --max_cluster_size
of 200 would skip. That parameter matters more than it might look: with
those two clusters excluded, the pairwise comparison count is 114,305
(96.3% under 2A); including them adds C(649,2) + C(373,2) = 210,276 +
69,378 = 279,654 pairs, for a total of 393,959. Because the pooled
statistic is sensitive to this parameter, --max_cluster_size should always
be reported alongside any pairwise RMSD number, and this script's default
recomputes the full set so its output is self-consistent end to end.

Reuses kabsch_rmsd() and isolate_h3_ca_coords() imported directly from
rq1_sequence_structural_bias.py -- the exact same superposition and
H3-isolation logic used elsewhere, so results cannot diverge from that
module's own definitions.

SUPERPOSITION METHOD (documented here precisely for the Methods text, by
reading the actual functions rather than describing them from memory):
  - Atoms: C-alpha only (one coordinate per residue; "coords" in the .pt
    file is indexed directly by residue, no separate atom axis).
  - Alignment: pairwise self-alignment (Kabsch/SVD) between the two
    structures being compared directly -- not alignment to a fixed
    reference frame or to the antibody framework region. Each pair is
    independently centered and rotated onto each other.
  - Only equal-H3-length pairs are compared; unequal-length pairs are
    counted separately and excluded (not padded or truncated).
  - Missing/unresolved coordinates: isolate_h3_ca_coords() drops a
    structure entirely from the cluster (not per-residue) if any NaN is
    present in its H3 coordinate block -- so partial models are excluded
    wholesale, not zero-filled or interpolated.
  - Insertion codes / altlocs: not handled explicitly at this stage;
    coords are already a fixed-length array per residue as extracted
    during preprocessing (Section 3.1), so any insertion-code or altloc
    resolution happened upstream, not in this RMSD step itself.
  - Single model only: exactly the model requested for that entry (no
    multi-model averaging or ensemble handling).
  - This is a C-alpha-only RMSD, not a full backbone RMSD (N/CA/C/O) --
    should be named "CDR-H3 Ca RMSD" throughout, not "backbone RMSD".

Usage
-----
    python scripts/audit_rmsd_weighting.py --config configs/config.yaml \
        --max_cluster_size 1000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

from rq1_sequence_structural_bias import kabsch_rmsd, isolate_h3_ca_coords


def compute_cluster_rmsds(group, pt_dir, master_lookup, collapse_vh_vl=False):
    """Loads H3 CA coords for every member of one cluster, optionally
    collapsing to one member per unique (heavy_seq, light_seq) pair first,
    and returns all pairwise equal-length RMSDs plus load/mismatch counts."""
    members = []
    seen_vh_vl = set()
    n_load_failures = 0
    for _, row in group.iterrows():
        stem = row["filename_stem"]
        if collapse_vh_vl:
            vh_vl = master_lookup.get(stem, {}).get("_vh_vl_key")
            if vh_vl is not None:
                if vh_vl in seen_vh_vl:
                    continue
                seen_vh_vl.add(vh_vl)
        pt_path = os.path.join(pt_dir, f"{stem}.pt")
        try:
            pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            n_load_failures += 1
            continue
        h3_ca = isolate_h3_ca_coords(pt_dict)
        if h3_ca is None:
            n_load_failures += 1
            continue
        members.append({"filename_stem": stem, "h3_len": h3_ca.shape[0], "h3_ca": h3_ca})

    rmsds = []
    n_length_mismatch = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            if a["h3_len"] != b["h3_len"]:
                n_length_mismatch += 1
                continue
            try:
                rmsds.append(kabsch_rmsd(a["h3_ca"], b["h3_ca"]))
            except ValueError:
                n_load_failures += 1
    return rmsds, len(members), n_length_mismatch, n_load_failures


def summarize(rmsds):
    if not rmsds:
        return None
    arr = np.array(rmsds)
    return {
        "n_pairs": len(arr), "mean_angstrom": float(arr.mean()),
        "median_angstrom": float(np.median(arr)), "std_angstrom": float(arr.std()),
        "fraction_under_2A": float((arr < 2.0).mean()),
        "fraction_under_1A": float((arr < 1.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max_cluster_size", type=int, default=1000,
                         help="Generous by design -- must exceed 649 (the largest known "
                              "cluster) to actually include it. Kabsch on ~10-20-residue "
                              "H3 arrays is cheap; even C(649,2)=210,276 pairs is fast.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    pt_dir = cfg["paths"]["sabdab_pt_dir"]

    clusters_path = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(clusters_path, "rq1_cdrh3_clusters.tsv")
    require_path(master_path, "master_antibodies.csv")

    clusters_df = pd.read_csv(clusters_path, sep="\t")
    master_df = pd.read_csv(master_path, low_memory=False)
    master_df["_vh_vl_key"] = list(zip(
        master_df["heavy_seq"].fillna(""), master_df["light_seq"].fillna("")
    ))
    master_lookup = master_df.set_index("filename_stem")[["_vh_vl_key"]].to_dict("index")

    cluster_groups = clusters_df.groupby("cluster_rep")
    non_singleton = {k: v for k, v in cluster_groups if len(v) > 1}
    log(f"{len(non_singleton)} non-singleton clusters")

    all_pairs_raw, all_pairs_collapsed = [], []
    per_cluster_raw, per_cluster_collapsed = [], []
    n_skipped_too_large = 0
    skipped = []

    for cluster_rep, group in non_singleton.items():
        if len(group) > args.max_cluster_size:
            n_skipped_too_large += 1
            skipped.append({"cluster_rep": cluster_rep, "n_members": len(group)})
            continue

        rmsds_raw, n_valid, n_mismatch, n_fail = compute_cluster_rmsds(
            group, pt_dir, master_lookup, collapse_vh_vl=False)
        if rmsds_raw:
            all_pairs_raw.extend(rmsds_raw)
            per_cluster_raw.append({
                "cluster_rep": cluster_rep, "n_members": len(group), "n_valid": n_valid,
                "n_pairs": len(rmsds_raw), "mean_rmsd": float(np.mean(rmsds_raw)),
                "fraction_under_2A": float((np.array(rmsds_raw) < 2.0).mean()),
            })

        rmsds_coll, n_valid_c, _, _ = compute_cluster_rmsds(
            group, pt_dir, master_lookup, collapse_vh_vl=True)
        if rmsds_coll:
            all_pairs_collapsed.extend(rmsds_coll)
            per_cluster_collapsed.append({
                "cluster_rep": cluster_rep, "n_members_after_collapse": n_valid_c,
                "n_pairs": len(rmsds_coll), "mean_rmsd": float(np.mean(rmsds_coll)),
                "fraction_under_2A": float((np.array(rmsds_coll) < 2.0).mean()),
            })

        if len(per_cluster_raw) % 500 == 0 and len(per_cluster_raw) > 0:
            log(f"  ...{len(per_cluster_raw)} clusters processed")

    log(f"Skipped {n_skipped_too_large} clusters exceeding max_cluster_size="
        f"{args.max_cluster_size}: {skipped}")

    # ── Pair-weighted (pooled) ──
    pair_weighted_raw = summarize(all_pairs_raw)
    pair_weighted_collapsed = summarize(all_pairs_collapsed)
    log(f"PAIR-WEIGHTED (raw): {pair_weighted_raw}")
    log(f"PAIR-WEIGHTED (VH/VL-collapsed): {pair_weighted_collapsed}")

    # ── Cluster-weighted (equal weight per cluster) ──
    def cluster_weighted_summary(per_cluster):
        if not per_cluster:
            return None
        fracs = np.array([c["fraction_under_2A"] for c in per_cluster])
        means = np.array([c["mean_rmsd"] for c in per_cluster])
        return {
            "n_clusters": len(per_cluster),
            "fraction_under_2A_mean_of_clusters": float(fracs.mean()),
            "fraction_under_2A_median_of_clusters": float(np.median(fracs)),
            "fraction_under_2A_std_of_clusters": float(fracs.std()),
            "mean_rmsd_mean_of_clusters": float(means.mean()),
            "mean_rmsd_median_of_clusters": float(np.median(means)),
        }

    cluster_weighted_raw = cluster_weighted_summary(per_cluster_raw)
    cluster_weighted_collapsed = cluster_weighted_summary(per_cluster_collapsed)
    log(f"CLUSTER-WEIGHTED (raw): {cluster_weighted_raw}")
    log(f"CLUSTER-WEIGHTED (VH/VL-collapsed): {cluster_weighted_collapsed}")

    # ── Two-largest-clusters contribution check (reproduces the professor's own math) ──
    two_largest = sorted(per_cluster_raw, key=lambda c: -c["n_members"])[:2]
    two_largest_pairs = sum(c["n_pairs"] for c in two_largest)
    total_pairs = sum(c["n_pairs"] for c in per_cluster_raw)
    log(f"Two largest clusters: {[(c['cluster_rep'], c['n_members'], c['n_pairs']) for c in two_largest]}")
    log(f"Their share of all pairs: {two_largest_pairs}/{total_pairs} = "
        f"{100*two_largest_pairs/total_pairs:.1f}%")

    report = {
        "reconciliation_note": (
            "Full recompute including all non-singleton clusters up to "
            f"max_cluster_size={args.max_cluster_size}, unlike the archived "
            "rq1_backbone_redundancy.json (max_cluster_size=200, silently excludes "
            "the 649- and 373-member clusters). See module docstring."
        ),
        "pair_weighted_raw": pair_weighted_raw,
        "pair_weighted_vh_vl_collapsed": pair_weighted_collapsed,
        "cluster_weighted_raw": cluster_weighted_raw,
        "cluster_weighted_vh_vl_collapsed": cluster_weighted_collapsed,
        "two_largest_clusters_pair_share_pct": round(100 * two_largest_pairs / total_pairs, 2) if total_pairs else None,
        "n_clusters_skipped_too_large": n_skipped_too_large,
        "clusters_skipped_too_large": skipped,
        "per_cluster_raw_all": per_cluster_raw,
    }
    out_path = os.path.join(work_dir, "tables", "rmsd_pair_vs_cluster_weighted.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
